"""Capture dashboard screenshots for the project report.

Starts the locally installed Google Chrome headless with a throwaway profile, signs in
through the API, selects a team, and saves each dashboard page as a PNG at 2× scale,
using the Chrome DevTools protocol (no extra browser download).

Usage (stack running, e.g. `docker compose up`):
    backend/.venv/bin/python docs/screenshots/capture.py [--base http://localhost:8080]
                                                         [--team "Demo (synthetic traffic)"]
"""

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402

OUT = Path(__file__).parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PAGES = [
    ("overview", "/", None),
    ("playground", "/playground", "playground"),
    ("requests", "/requests", "first_request"),
    ("forecast", "/forecast", None),
    ("alerts", "/alerts", None),
    ("optimization", "/optimization", None),
    ("whatif", "/whatif", "simulate"),
    ("models", "/models", None),
    ("teams", "/teams", None),
]


class Tab:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def send(self, method: str, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr: str):
        r = await self.send("Runtime.evaluate", expression=expr, awaitPromise=True,
                            returnByValue=True)
        return r.get("result", {}).get("value")

    async def goto(self, url: str, settle: float = 3.5):
        await self.send("Page.navigate", url=url)
        await asyncio.sleep(settle)

    async def click_text(self, text: str, selector: str = "button") -> bool:
        return await self.js(f"""(() => {{
            const el = [...document.querySelectorAll({json.dumps(selector)})]
              .find(e => e.textContent.trim().includes({json.dumps(text)}));
            if (el) el.click();
            return Boolean(el);
        }})()""")

    async def shot(self, path: Path) -> None:
        height = await self.js("document.documentElement.scrollHeight")
        await self.send("Emulation.setDeviceMetricsOverride", width=1440,
                        height=min(int(height), 2600), deviceScaleFactor=2, mobile=False)
        await asyncio.sleep(1.0)
        r = await self.send("Page.captureScreenshot", format="png", captureBeyondViewport=True)
        path.write_bytes(base64.b64decode(r["data"]))
        await self.send("Emulation.setDeviceMetricsOverride", width=1440, height=900,
                        deviceScaleFactor=2, mobile=False)


async def capture(base: str, team: str) -> None:
    s = get_settings()
    tokens = httpx.post(f"{base}/api/v1/auth/login", data={
        "username": s.seed_admin_email, "password": s.seed_admin_password}).json()
    teams = httpx.get(f"{base}/api/v1/teams",
                      headers={"Authorization": f"Bearer {tokens['access_token']}"}).json()
    team_id = next(t["id"] for t in teams if t["name"] == team)

    profile = tempfile.mkdtemp(prefix="budgetopt-shots-")
    port = 9333
    proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}",
                             f"--user-data-dir={profile}", "--no-first-run",
                             "--hide-scrollbars", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                targets = httpx.get(f"http://127.0.0.1:{port}/json").json()
                page = next(t for t in targets if t["type"] == "page")
                break
            except (httpx.HTTPError, StopIteration):
                time.sleep(0.2)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            tab = Tab(ws)
            await tab.send("Page.enable")
            await tab.send("Emulation.setDeviceMetricsOverride", width=1440, height=900,
                           deviceScaleFactor=2, mobile=False)
            await tab.goto(f"{base}/", settle=1.5)
            await tab.js(f"""localStorage.setItem('budgetopt.access', {json.dumps(tokens['access_token'])});
                localStorage.setItem('budgetopt.refresh', {json.dumps(tokens['refresh_token'])});
                localStorage.setItem('budgetopt.team', '{team_id}');""")
            for name, path, action in PAGES:
                await tab.goto(f"{base}{path}")
                if action == "playground":
                    await tab.click_text("Send")
                    await asyncio.sleep(10)
                elif action == "simulate":
                    await tab.click_text("Simulate")
                    await asyncio.sleep(5)
                elif action == "first_request":
                    await tab.js("document.querySelector('tbody tr')?.click()")
                    await asyncio.sleep(2.5)
                await tab.shot(OUT / f"{name}.png")
                print("saved", name)
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--team", default="Demo (synthetic traffic)")
    args = ap.parse_args()
    asyncio.run(capture(args.base, args.team))

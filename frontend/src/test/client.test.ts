import { api, ApiError, onLogout, tokens } from "../api/client";

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("api client", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("sends the bearer token and parses JSON", async () => {
    tokens.set("a1", "r1");
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(json(200, { ok: true }));
    await expect(api("/teams")).resolves.toEqual({ ok: true });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/teams");
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer a1");
  });

  it("refreshes an expired access token once and retries", async () => {
    tokens.set("expired", "r1");
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(json(401, { detail: "expired" }))
      .mockResolvedValueOnce(json(200, { access_token: "a2", refresh_token: "r2" }))
      .mockResolvedValueOnce(json(200, { id: 1 }));
    await expect(api("/auth/me")).resolves.toEqual({ id: 1 });
    expect(fetchMock.mock.calls[1][0]).toBe("/api/v1/auth/refresh");
    expect(new Headers(fetchMock.mock.calls[2][1]?.headers).get("Authorization")).toBe("Bearer a2");
    expect(tokens.access).toBe("a2");
  });

  it("signs out when the refresh token is rejected", async () => {
    tokens.set("expired", "bad");
    const loggedOut = vi.fn();
    const off = onLogout(loggedOut);
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(json(401, { detail: "expired" }))
      .mockResolvedValueOnce(json(401, { detail: "bad refresh" }));
    await expect(api("/auth/me")).rejects.toBeInstanceOf(ApiError);
    expect(loggedOut).toHaveBeenCalled();
    expect(tokens.access).toBeNull();
    off();
  });

  it("surfaces FastAPI and OpenAI-style error messages", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(json(403, { detail: "You do not have access to this team" }))
      .mockResolvedValueOnce(json(402, { error: { message: "budget exceeded" } }));
    await expect(api("/teams/2/spend")).rejects.toThrow("You do not have access to this team");
    await expect(api("/route")).rejects.toThrow("budget exceeded");
  });
});

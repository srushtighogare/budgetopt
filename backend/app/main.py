from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1 import auth, gateway, models, profiling, providers, requests, teams, users
from app.core.config import get_settings
from app.services.workload.analyzer import warm_up

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    warm_up()  # load ML models before the first request
    yield


app = FastAPI(
    lifespan=lifespan,
    title="BudgetOpt API",
    description="AI based cost-optimal LLM selection and token budget optimization platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (auth.router, users.router, teams.router, models.router, providers.router,
          requests.router, gateway.router, profiling.router):
    app.include_router(r, prefix="/api/v1")
app.include_router(gateway.openai_router)


def _is_openai_path(request: Request) -> bool:
    return request.url.path.startswith("/v1/")


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    """OpenAI-compatible clients expect {"error": {...}} bodies on /v1 routes."""
    if _is_openai_path(request):
        code = {401: "invalid_api_key", 404: "not_found"}.get(exc.status_code, "error")
        return gateway.openai_error(exc.status_code, str(exc.detail), code, headers=exc.headers)
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    if _is_openai_path(request):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return gateway.openai_error(400, f"{loc}: {first.get('msg', 'invalid request')}",
                                    "invalid_request_error", param=loc or None)
    return await request_validation_exception_handler(request, exc)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}

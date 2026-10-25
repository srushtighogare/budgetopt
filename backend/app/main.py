from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import auth, models, providers, teams, users
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
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

for r in (auth.router, users.router, teams.router, models.router, providers.router):
    app.include_router(r, prefix="/api/v1")


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}

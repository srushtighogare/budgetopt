"""Request log with routing explanations (report: explainability / audit trail)."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_accessible_team, get_current_user
from app.db.models import LLMModel, Request, RequestStatus, Role, RoutingDecision, Team, User
from app.db.session import get_db
from app.schemas.gateway import RequestDetailOut, RequestLogOut

router = APIRouter(prefix="/requests", tags=["requests"])


def _base() -> Select:
    return (
        select(Request, LLMModel.name, RoutingDecision)
        .join(Team, Team.id == Request.team_id)
        .outerjoin(LLMModel, LLMModel.id == Request.model_id)
        .outerjoin(RoutingDecision, RoutingDecision.request_id == Request.id)
    )


def _scope(q: Select, user: User) -> Select:
    """Members see their own requests, managers their team's, admins their org's."""
    q = q.where(Team.organization_id == user.organization_id)
    if user.role == Role.MEMBER:
        return q.where(Request.user_id == user.id)
    if user.role == Role.TEAM_MANAGER:
        return q.where(Request.team_id == user.team_id)
    return q


def _row(r: Request, model: str | None, d: RoutingDecision | None) -> dict:
    return {
        "id": r.id, "team_id": r.team_id, "user_id": r.user_id, "api_key_id": r.api_key_id,
        "status": r.status.value, "model": model, "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens, "cost": r.cost, "latency_ms": r.latency_ms,
        "created_at": r.created_at, "explanation": d.explanation if d else None,
        "savings": d.savings if d else None,
    }


@router.get("", response_model=list[RequestLogOut])
async def list_requests(
    team_id: int | None = None,
    status_filter: RequestStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
) -> list[dict]:
    q = _scope(_base(), user)
    if team_id is not None:
        await get_accessible_team(team_id, user, db, action="list_requests")
        q = q.where(Request.team_id == team_id)
    if status_filter is not None:
        q = q.where(Request.status == status_filter)
    q = q.order_by(Request.created_at.desc(), Request.id.desc()).limit(limit).offset(offset)
    return [_row(*row) for row in (await db.execute(q)).all()]


@router.get("/{request_id}", response_model=RequestDetailOut)
async def get_request(
    request_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> dict:
    row = (await db.execute(_scope(_base(), user).where(Request.id == request_id))).first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Request not found")
    r, model, d = row
    return {
        **_row(r, model, d), "prompt_text": r.prompt_text,
        "est_input_tokens": r.est_input_tokens, "est_output_tokens": r.est_output_tokens,
        "budget_state": d.budget_state if d else None,
        "baseline_cost": d.baseline_cost if d else None,
        "fallback_used": d.fallback_used if d else None,
        "candidates": d.candidates if d else None,
    }

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_accessible_team, get_current_user, require_admin, require_manager
from app.core.security import generate_api_key
from app.db.models import ApiKey, AuditEvent, Role, Team, User
from app.db.session import get_db
from app.schemas.common import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyOut,
    TeamCreate,
    TeamOut,
    TeamSpendOut,
    TeamUpdate,
)
from app.services.budget import budget_status

router = APIRouter(prefix="/teams", tags=["teams"])


@router.get("", response_model=list[TeamOut])
async def list_teams(
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> list[Team]:
    q = select(Team).where(Team.organization_id == user.organization_id).order_by(Team.id)
    if user.role != Role.ORG_ADMIN:
        q = q.where(Team.id == user.team_id)
    return list(await db.scalars(q))


@router.post("", response_model=TeamOut, status_code=status.HTTP_201_CREATED)
async def create_team(
    body: TeamCreate, user: User = Depends(require_admin), db: AsyncSession = Depends(get_db)
) -> Team:
    exists = await db.scalar(
        select(Team.id).where(Team.organization_id == user.organization_id, Team.name == body.name)
    )
    if exists:
        raise HTTPException(status.HTTP_409_CONFLICT, "Team name already exists")
    team = Team(organization_id=user.organization_id, **body.model_dump())
    db.add(team)
    await db.flush()
    db.add(AuditEvent(user_id=user.id, action="create_team", resource="team",
                      resource_id=str(team.id),
                      details={"monthly_budget": str(body.monthly_budget)}))
    await db.commit()
    return team


@router.get("/{team_id}", response_model=TeamOut)
async def get_team(
    team_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Team:
    return await get_accessible_team(team_id, user, db)


@router.patch("/{team_id}", response_model=TeamOut)
async def update_team(
    team_id: int, body: TeamUpdate,
    user: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> Team:
    team = await get_accessible_team(team_id, user, db, action="update_team")
    changes = body.model_dump(exclude_unset=True)
    for field, value in changes.items():
        setattr(team, field, value)
    db.add(AuditEvent(user_id=user.id, action="update_team", resource="team",
                      resource_id=str(team.id),
                      details={k: str(v) for k, v in changes.items()}))
    await db.commit()
    return team


@router.get("/{team_id}/spend", response_model=TeamSpendOut)
async def get_team_spend(
    team_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> TeamSpendOut:
    team = await get_accessible_team(team_id, user, db, action="view_spend")
    s = await budget_status(db, team)
    db.add(AuditEvent(user_id=user.id, action="view_spend", resource="team",
                      resource_id=str(team.id)))
    await db.commit()
    return TeamSpendOut(
        team_id=team.id, period_start=s.period_start, monthly_budget=s.monthly_budget,
        spend=s.spend, usage_ratio=s.usage_ratio, budget_state=s.state.value,
    )


# --- API keys for applications using the gateway ---


@router.get("/{team_id}/api-keys", response_model=list[ApiKeyOut])
async def list_api_keys(
    team_id: int, user: User = Depends(require_manager), db: AsyncSession = Depends(get_db)
) -> list[ApiKey]:
    await get_accessible_team(team_id, user, db, action="list_api_keys")
    return list(await db.scalars(select(ApiKey).where(ApiKey.team_id == team_id)))


@router.post("/{team_id}/api-keys", response_model=ApiKeyCreated,
             status_code=status.HTTP_201_CREATED)
async def create_api_key(
    team_id: int, body: ApiKeyCreate,
    user: User = Depends(require_manager), db: AsyncSession = Depends(get_db),
) -> ApiKeyCreated:
    await get_accessible_team(team_id, user, db, action="create_api_key")
    plaintext, prefix, key_hash = generate_api_key()
    key = ApiKey(team_id=team_id, created_by=user.id, name=body.name,
                 prefix=prefix, key_hash=key_hash)
    db.add(key)
    await db.flush()
    db.add(AuditEvent(user_id=user.id, action="create_api_key", resource="api_key",
                      resource_id=str(key.id)))
    await db.commit()
    return ApiKeyCreated(**ApiKeyOut.model_validate(key).model_dump(), key=plaintext)


@router.delete("/{team_id}/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    team_id: int, key_id: int,
    user: User = Depends(require_manager), db: AsyncSession = Depends(get_db),
) -> None:
    await get_accessible_team(team_id, user, db, action="revoke_api_key")
    key = await db.get(ApiKey, key_id)
    if key is None or key.team_id != team_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "API key not found")
    key.is_active = False  # revoke, keep the row for audit history
    db.add(AuditEvent(user_id=user.id, action="revoke_api_key", resource="api_key",
                      resource_id=str(key.id),
                      details={"revoked_at": datetime.now(UTC).isoformat()}))
    await db.commit()

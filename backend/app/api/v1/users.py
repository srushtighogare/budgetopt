from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_accessible_team, require_manager
from app.core.security import hash_password
from app.db.models import AuditEvent, Role, User
from app.db.session import get_db
from app.schemas.common import UserCreate, UserOut

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[UserOut])
async def list_users(
    user: User = Depends(require_manager), db: AsyncSession = Depends(get_db)
) -> list[User]:
    q = select(User).where(User.organization_id == user.organization_id).order_by(User.id)
    if user.role != Role.ORG_ADMIN:
        q = q.where(User.team_id == user.team_id)
    return list(await db.scalars(q))


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate, user: User = Depends(require_manager), db: AsyncSession = Depends(get_db)
) -> User:
    # Managers may only add members to their own team
    if user.role != Role.ORG_ADMIN and (body.role != Role.MEMBER or body.team_id != user.team_id):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Managers can only add team members")
    if body.team_id is not None:
        await get_accessible_team(body.team_id, user, db, action="create_user")
    elif body.role != Role.ORG_ADMIN:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Non-admin users need a team")

    email = body.email.lower()
    if await db.scalar(select(User.id).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")

    new = User(
        organization_id=user.organization_id, team_id=body.team_id, email=email,
        full_name=body.full_name, password_hash=hash_password(body.password), role=body.role,
    )
    db.add(new)
    await db.flush()
    db.add(AuditEvent(user_id=user.id, action="create_user", resource="user",
                      resource_id=str(new.id), details={"role": body.role.value}))
    await db.commit()
    return new

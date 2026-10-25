"""FastAPI dependencies: authentication and role-based access control.

Access rules (report, "Safety and Security"):
- member:        own team's data only, own requests only
- team_manager:  everything belonging to their team
- org_admin:     everything in their organization
Denied access attempts are recorded in `audit_events`.
"""

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import API_KEY_PREFIX, decode_token, hash_api_key
from app.db.models import ApiKey, AuditEvent, Role, Team, User
from app.db.session import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)
) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token, "access")
        user_id = int(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError):
        raise unauthorized from None

    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        raise unauthorized
    return user


def require_roles(*roles: Role) -> Callable[..., Coroutine[Any, Any, User]]:
    async def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient role")
        return user

    return checker


require_admin = require_roles(Role.ORG_ADMIN)
require_manager = require_roles(Role.ORG_ADMIN, Role.TEAM_MANAGER)


def can_access_team(user: User, team: Team) -> bool:
    if user.organization_id != team.organization_id:
        return False
    if user.role == Role.ORG_ADMIN:
        return True
    return user.team_id == team.id


async def get_accessible_team(
    team_id: int, user: User, db: AsyncSession, action: str = "view_team"
) -> Team:
    """Load a team, enforcing that `user` may see it. Denials are audited.

    Returns 404 (not 403) for other organizations' teams so ids can't be probed.
    """
    team = await db.get(Team, team_id)
    if team is None or team.organization_id != user.organization_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Team not found")
    if not can_access_team(user, team):
        db.add(
            AuditEvent(
                user_id=user.id, action=action, resource="team",
                resource_id=str(team_id), allowed=False,
            )
        )
        await db.commit()
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You do not have access to this team")
    return team


# --- Gateway callers (applications using an API key) ---

api_key_scheme = HTTPBearer(auto_error=False)


async def get_api_key(
    creds: HTTPAuthorizationCredentials | None = Depends(api_key_scheme),
    db: AsyncSession = Depends(get_db),
) -> ApiKey:
    if creds is None or not creds.credentials.startswith(API_KEY_PREFIX):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid API key",
                            headers={"WWW-Authenticate": "Bearer"})
    key = await db.scalar(select(ApiKey).where(ApiKey.key_hash == hash_api_key(creds.credentials)))
    if key is None or not key.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid API key",
                            headers={"WWW-Authenticate": "Bearer"})
    key.last_used_at = datetime.now(UTC)
    await db.commit()
    return key

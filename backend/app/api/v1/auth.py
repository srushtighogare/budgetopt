import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.core.security import create_token, decode_token, verify_password
from app.db.models import User
from app.db.session import get_db
from app.schemas.common import RefreshIn, TokenPair, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


def _issue(user: User) -> TokenPair:
    claims = {"role": user.role.value, "org": user.organization_id}
    return TokenPair(
        access_token=create_token(str(user.id), "access", claims),
        refresh_token=create_token(str(user.id), "refresh"),
    )


@router.post("/login", response_model=TokenPair)
async def login(
    form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)
) -> TokenPair:
    user = await db.scalar(select(User).where(User.email == form.username.lower()))
    # Same error for unknown email and wrong password (no account enumeration)
    if user is None or not user.is_active or not verify_password(form.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return _issue(user)


@router.post("/refresh", response_model=TokenPair)
async def refresh(body: RefreshIn, db: AsyncSession = Depends(get_db)) -> TokenPair:
    try:
        user_id = int(decode_token(body.refresh_token, "refresh")["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token") from None
    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token")
    return _issue(user)


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)) -> User:
    return user

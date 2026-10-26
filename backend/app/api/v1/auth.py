import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.core.redis import get_redis
from app.core.security import create_token, decode_token, hash_password, verify_password
from app.db.models import User
from app.db.session import get_db
from app.schemas.common import RefreshIn, TokenPair, UserOut
from app.services import login_guard

router = APIRouter(prefix="/auth", tags=["auth"])


def _issue(user: User) -> TokenPair:
    claims = {"role": user.role.value, "org": user.organization_id}
    return TokenPair(
        access_token=create_token(str(user.id), "access", claims),
        refresh_token=create_token(str(user.id), "refresh"),
    )


# Verified against when the email is unknown, so both failure paths cost the same argon2
# work and response timing doesn't reveal which emails have accounts
_DUMMY_HASH = hash_password("timing-equaliser-not-a-real-password-1")


@router.post("/login", response_model=TokenPair)
async def login(
    request: Request, form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db), redis: Redis = Depends(get_redis),
) -> TokenPair:
    email = form.username.lower()
    ip = request.client.host if request.client else "unknown"
    verdict = await login_guard.check(redis, email, ip)
    if not verdict.allowed:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            "Too many failed sign-in attempts. Try again later.",
                            headers={"Retry-After": str(verdict.retry_after_s)})

    user = await db.scalar(select(User).where(User.email == email))
    password_ok = verify_password(form.password, user.password_hash if user else _DUMMY_HASH)
    # Same error for unknown email and wrong password (no account enumeration)
    if user is None or not user.is_active or not password_ok:
        await login_guard.record_failure(redis, email, ip)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    await login_guard.record_success(redis, email, ip)
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

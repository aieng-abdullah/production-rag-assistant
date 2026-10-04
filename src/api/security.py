"""JWT issue/verify — HS256, env-only secret (PLAN.md PR-2b, security gate 3).

Framework-free: no FastAPI imports, safe to call from any layer.
Tokens are short-lived claims (`sub`, `exp`); the secret never leaves env.
"""

from datetime import datetime, timedelta, timezone

import jwt

from src.config import Config

__all__ = ["create_token", "decode_token", "TokenError"]


class TokenError(Exception):
    """Raised when a token is missing claims, malformed, tampered, or expired."""


def _secret() -> str:
    secret = Config.JWT_SECRET
    if not secret:
        raise EnvironmentError(
            "JWT_SECRET is not set. Add it to .env to enable signed tokens."
        )
    return secret


def create_token(user_id: int, *, ttl_days: int | None = None) -> str:
    """Sign a bearer token for `user_id` (7-day default)."""
    ttl = Config.JWT_TTL_DAYS if ttl_days is None else ttl_days
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(days=ttl),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _secret(), algorithm="HS256")


def decode_token(token: str) -> int:
    """Verify signature + expiry, return `user_id`. Raises `TokenError` on failure."""
    try:
        payload = jwt.decode(token, _secret(), algorithms=["HS256"])
        return int(payload["sub"])
    except jwt.PyJWTError as exc:
        raise TokenError("invalid or expired token") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise TokenError("token missing user claim") from exc

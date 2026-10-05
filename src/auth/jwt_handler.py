"""JWT token handling for authentication."""

from datetime import datetime, timedelta, timezone

import jwt
from loguru import logger

from src.config import Config


ALGORITHM = "HS256"
TOKEN_EXPIRY_DAYS = 7


def _get_secret() -> str:
    """Get JWT secret from config."""
    secret = Config.JWT_SECRET_KEY
    if not secret:
        raise EnvironmentError(
            "JWT_SECRET_KEY not set. Generate with: openssl rand -hex 32"
        )
    return secret


def create_token(user_id: int, email: str, tier: str) -> str:
    """Create a JWT token for the user."""
    now = datetime.now(timezone.utc)
    payload = {
        "user_id": user_id,
        "email": email,
        "tier": tier,
        "iat": now,
        "exp": now + timedelta(days=TOKEN_EXPIRY_DAYS),
    }
    token = jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)
    logger.debug(f"Created token for user_id={user_id} tier={tier}")
    return token


def verify_token(token: str) -> dict:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, _get_secret(), algorithms=[ALGORITHM])
        logger.debug(f"Verified token for user_id={payload.get('user_id')}")
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        raise
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise
"""FastAPI dependencies (PLAN.md PR-2b/PR-3)."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.security import TokenError, decode_token
from src.services.quotas import QuotaExceeded

__all__ = ["require_user", "quota_to_http"]

_bearer = HTTPBearer(auto_error=False)


def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> int:
    """Resolve the authenticated `user_id` from the Bearer token.

    401 on absent/malformed/expired token. Callers thread the returned
    id as `tenant_id` into RAGService.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_token(credentials.credentials)
    except TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except EnvironmentError as exc:
        # JWT_SECRET missing — server misconfiguration, fail loudly as 500.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


def quota_to_http(exc: QuotaExceeded) -> HTTPException:
    """429 with a clear message; `Retry-After` when the quota resets daily."""
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=str(exc),
        headers=headers,
    )

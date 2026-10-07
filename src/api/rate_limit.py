"""Per-IP sliding-window rate limiting (in-memory, no new dependencies).

Route classes: GLOBAL 60/min for everything, AUTH 5/min on `/auth/*`,
ANON 3/hour on `POST /auth/anonymous` — the guest-identity farming hole,
since that endpoint mints a fresh quota per submitted device id. Keys are
`(route_class, ip)`; the store is bounded by MAX_TRACKED_KEYS with
opportunistic eviction so an attacker cannot grow it without limit.
"""

import math
import time
from collections import deque

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

__all__ = [
    "ANON",
    "ANON_LIMIT",
    "ANON_WINDOW_SECONDS",
    "AUTH",
    "AUTH_LIMIT",
    "AUTH_WINDOW_SECONDS",
    "GLOBAL",
    "GLOBAL_LIMIT",
    "GLOBAL_WINDOW_SECONDS",
    "MAX_TRACKED_KEYS",
    "POLICIES",
    "RateLimitMiddleware",
    "SlidingWindow",
    "route_class",
]

GLOBAL = "global"
AUTH = "auth"
ANON = "anon"

GLOBAL_LIMIT = 60
GLOBAL_WINDOW_SECONDS = 60.0
AUTH_LIMIT = 5
AUTH_WINDOW_SECONDS = 60.0
ANON_LIMIT = 3
ANON_WINDOW_SECONDS = 3600.0
MAX_TRACKED_KEYS = 10_000

POLICIES: dict[str, tuple[int, float]] = {
    GLOBAL: (GLOBAL_LIMIT, GLOBAL_WINDOW_SECONDS),
    AUTH: (AUTH_LIMIT, AUTH_WINDOW_SECONDS),
    ANON: (ANON_LIMIT, ANON_WINDOW_SECONDS),
}


def _now() -> float:
    """Monotonic seconds — module-level so tests can monkeypatch the clock."""
    return time.monotonic()


def route_class(method: str, path: str) -> str | None:
    """Policy bucket for a request; `None` means exempt."""
    if method.upper() == "OPTIONS" or path == "/health":
        return None
    if method.upper() == "POST" and path.rstrip("/") == "/auth/anonymous":
        return ANON
    if path == "/auth" or path.startswith("/auth/"):
        return AUTH
    return GLOBAL


class SlidingWindow:
    """Bounded per-key counter of hit timestamps inside the window."""

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = {}

    def hit(self, key: tuple[str, str], limit: int, window: float) -> int | None:
        """Record a hit. Returns `Retry-After` seconds when the limit is
        already reached (the hit is NOT recorded), else `None`."""
        now = _now()
        hits = self._hits.get(key)
        if hits is None:
            if len(self._hits) >= MAX_TRACKED_KEYS:
                self._evict(now)
            hits = deque()
            self._hits[key] = hits
        cutoff = now - window
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= limit:
            return max(1, math.ceil(hits[0] + window - now))
        hits.append(now)
        return None

    def key_count(self) -> int:
        return len(self._hits)

    def _evict(self, now: float) -> None:
        for key in list(self._hits):
            hits = self._hits[key]
            window = POLICIES[key[0]][1]
            if not hits or hits[-1] <= now - window:
                del self._hits[key]
        while len(self._hits) >= MAX_TRACKED_KEYS:
            self._hits.pop(next(iter(self._hits)))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rejects over-quota requests with 429 + `Retry-After` before routing."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._window = SlidingWindow()

    async def dispatch(self, request: Request, call_next):
        route = route_class(request.method, request.url.path)
        if route is None:
            return await call_next(request)
        # Direct-uvicorn single instance: client.host only, X-Forwarded-For trust is a proxy follow-up.
        client = request.client
        ip = client.host if client else "unknown"
        limit, window = POLICIES[route]
        retry_after = self._window.hit((route, ip), limit, window)
        if retry_after is None:
            return await call_next(request)
        logger.warning(
            "Rate limit exceeded route_class={route_class} ip={ip} path={path}",
            route_class=route,
            ip=ip,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=429,
            content={"detail": f"Too many requests. Try again in {retry_after}s."},
            headers={"Retry-After": str(retry_after)},
        )

"""Process-local request rate limits and body limits for the demo API."""

import asyncio
from collections import defaultdict, deque
from math import ceil
from threading import Lock
from time import monotonic

from fastapi import HTTPException
from fastapi.responses import JSONResponse

RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_POLICIES = {
    ("POST", "/upload-file"): ("upload", 5),
    ("POST", "/query"): ("query", 20),
    ("POST", "/query/stream"): ("query", 20),
    ("GET", "/documents"): ("read", 120),
    ("DELETE", "/database"): ("admin", 10),
    ("DELETE", "/documents/{document_id}"): ("admin", 10),
    ("POST", "/documents/{document_id}/replace"): ("upload", 5),
    ("DELETE", "/jobs/{job_id}"): ("admin", 10),
    ("GET", "/database"): ("read", 120),
    ("GET", "/jobs/{job_id}"): ("read", 120),
    ("GET", "/jobs/{job_id}/stream"): ("read", 120),
}
_BODY_LIMITS = {
    ("POST", "/upload-file"): (20 * 1024 * 1024 + 64 * 1024, 300),
    ("POST", "/query"): (64 * 1024, 30),
    ("POST", "/query/stream"): (64 * 1024, 30),
    ("POST", "/documents/{document_id}/replace"): (
        20 * 1024 * 1024 + 64 * 1024,
        300,
    ),
    ("DELETE", "/database"): (8 * 1024, 30),
}


class RateLimiter:
    def __init__(self, window_seconds=RATE_LIMIT_WINDOW_SECONDS):
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
        self.lock = Lock()
        self.last_cleanup = monotonic()

    def check(self, client_ip: str, category: str, limit: int, now=None):
        now = monotonic() if now is None else now
        key = (client_ip, category)
        with self.lock:
            bucket = self.requests[key]
            cutoff = now - self.window_seconds
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                return max(1, ceil(bucket[0] + self.window_seconds - now))
            bucket.append(now)
            if (
                len(self.requests) > 10_000
                and now - self.last_cleanup >= self.window_seconds
            ):
                stale_before = cutoff
                stale_keys = [
                    bucket_key
                    for bucket_key, timestamps in self.requests.items()
                    if not timestamps or timestamps[-1] <= stale_before
                ]
                for stale_key in stale_keys:
                    self.requests.pop(stale_key, None)
                self.last_cleanup = now
        return None

    def reset(self):
        with self.lock:
            self.requests.clear()
            self.last_cleanup = monotonic()


class RateLimitMiddleware:
    def __init__(self, app, limiter: RateLimiter):
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            policy = self._policy(scope["method"], scope["path"])
            if policy:
                category, limit = policy
                client = scope.get("client")
                client_ip = str(client[0]) if client else "unknown"
                retry_after = self.limiter.check(client_ip, category, limit)
                if retry_after is not None:
                    response = JSONResponse(
                        status_code=429,
                        content={"detail": "Too many requests. Try again later."},
                        headers={"Retry-After": str(retry_after)},
                    )
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)

    @staticmethod
    def _policy(method: str, path: str):
        path = path.rstrip("/") or "/"
        exact = _RATE_POLICIES.get((method, path))
        if exact:
            return exact
        segments = path.strip("/").split("/")
        if len(segments) == 2 and segments[0] == "jobs":
            return _RATE_POLICIES.get((method, "/jobs/{job_id}"))
        if len(segments) == 3 and segments[0] == "jobs" and segments[2] == "stream":
            return _RATE_POLICIES.get((method, "/jobs/{job_id}/stream"))
        if len(segments) == 2 and segments[0] == "documents":
            return _RATE_POLICIES.get((method, "/documents/{document_id}"))
        if (
            len(segments) == 3
            and segments[0] == "documents"
            and segments[2] == "replace"
        ):
            return _RATE_POLICIES.get((method, "/documents/{document_id}/replace"))
        return None


class RequestLimitsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limits = self._limits(scope["method"], scope["path"])
        if limits is None:
            await self.app(scope, receive, send)
            return

        max_bytes, timeout_seconds = limits
        content_length = self._content_length(scope.get("headers", []))
        if content_length is not None and content_length > max_bytes:
            await JSONResponse(
                status_code=413,
                content={"detail": "Request body is too large."},
            )(scope, receive, send)
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        bytes_received = 0

        async def receive_with_limits():
            nonlocal bytes_received
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise HTTPException(
                    status_code=408,
                    detail="Request body was not received in time.",
                )
            try:
                message = await asyncio.wait_for(receive(), timeout=remaining)
            except TimeoutError as exc:
                raise HTTPException(
                    status_code=408,
                    detail="Request body was not received in time.",
                ) from exc
            if message["type"] == "http.request":
                bytes_received += len(message.get("body", b""))
                if bytes_received > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Request body is too large.",
                    )
            return message

        await self.app(scope, receive_with_limits, send)

    @staticmethod
    def _limits(method: str, path: str):
        path = path.rstrip("/") or "/"
        limits = _BODY_LIMITS.get((method, path))
        if limits:
            return limits
        segments = path.strip("/").split("/")
        if method == "DELETE" and len(segments) == 2 and segments[0] == "jobs":
            return 8 * 1024, 30
        if (
            method == "POST"
            and len(segments) == 3
            and segments[0] == "documents"
            and segments[2] == "replace"
        ):
            return _BODY_LIMITS.get((method, "/documents/{document_id}/replace"))
        return None

    @staticmethod
    def _content_length(headers):
        for name, value in headers:
            if name.lower() == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    return None
        return None

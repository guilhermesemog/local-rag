import asyncio

from fastapi import FastAPI, Request

from security import RateLimiter, RateLimitMiddleware, RequestLimitsMiddleware


def test_rate_limiter_applies_a_sliding_window_per_client_and_category():
    limiter = RateLimiter(window_seconds=60)

    assert limiter.check("192.0.2.1", "query", 2, now=0) is None
    assert limiter.check("192.0.2.1", "query", 2, now=10) is None
    assert limiter.check("192.0.2.1", "query", 2, now=20) == 40
    assert limiter.check("192.0.2.2", "query", 2, now=20) is None
    assert limiter.check("192.0.2.1", "upload", 1, now=20) is None
    assert limiter.check("192.0.2.1", "query", 2, now=60) is None


def test_rate_limiter_reset_clears_previous_requests():
    limiter = RateLimiter(window_seconds=60)
    limiter.check("192.0.2.1", "query", 1, now=0)
    assert limiter.check("192.0.2.1", "query", 1, now=1) == 59

    limiter.reset()

    assert limiter.check("192.0.2.1", "query", 1, now=1) is None


def test_rate_limit_middleware_maps_protected_routes_to_policies():
    assert RateLimitMiddleware._policy("POST", "/upload-file") == ("upload", 5)
    assert RateLimitMiddleware._policy("POST", "/upload-file/") == ("upload", 5)
    assert RateLimitMiddleware._policy("POST", "/query") == ("query", 20)
    assert RateLimitMiddleware._policy("POST", "/query/stream") == ("query", 20)
    assert RateLimitMiddleware._policy("GET", "/documents") == ("read", 120)
    assert RateLimitMiddleware._policy("DELETE", "/documents/doc-1") == ("admin", 10)
    assert RateLimitMiddleware._policy("POST", "/documents/doc-1/replace") == (
        "upload",
        5,
    )
    assert RateLimitMiddleware._policy("GET", "/jobs/abc123") == ("read", 120)
    assert RateLimitMiddleware._policy("GET", "/jobs/abc123/stream") == ("read", 120)
    assert RateLimitMiddleware._policy("DELETE", "/database") == ("admin", 10)
    assert RateLimitMiddleware._policy("GET", "/health") is None


def test_rate_limit_middleware_returns_429_and_retry_after():
    responses = []

    async def app(_scope, _receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        responses.append(message)

    middleware = RateLimitMiddleware(app, RateLimiter())
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/query",
        "client": ("192.0.2.1", 5000),
        "headers": [(b"x-forwarded-for", b"198.51.100.99")],
    }

    async def run():
        for _ in range(20):
            await middleware(scope, receive, send)
        await middleware(scope, receive, send)

    asyncio.run(run())

    denied = responses[-2]
    assert denied["type"] == "http.response.start"
    assert denied["status"] == 429
    headers = dict(denied["headers"])
    assert headers[b"retry-after"] == b"60"


def test_request_limits_bound_upload_and_query_body_sizes():
    assert RequestLimitsMiddleware._limits("POST", "/upload-file") == (
        20 * 1024 * 1024 + 64 * 1024,
        300,
    )
    assert RequestLimitsMiddleware._limits("POST", "/query") == (64 * 1024, 30)
    assert RequestLimitsMiddleware._limits("POST", "/query/stream") == (64 * 1024, 30)
    assert RequestLimitsMiddleware._limits("POST", "/documents/doc-1/replace") == (
        20 * 1024 * 1024 + 64 * 1024,
        300,
    )
    assert RequestLimitsMiddleware._limits("GET", "/health") is None


def test_request_limits_reject_a_large_body_before_route_handling():
    middleware = RequestLimitsMiddleware(lambda *_args: None)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/query",
        "headers": [(b"content-length", b"65537")],
    }
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    assert sent[0]["status"] == 413


def test_request_limits_reject_chunked_bodies_over_the_limit(monkeypatch):
    monkeypatch.setattr(
        RequestLimitsMiddleware, "_limits", staticmethod(lambda *_args: (4, 1))
    )

    fastapi_app = FastAPI()

    @fastapi_app.post("/query")
    async def read_body(request: Request):
        await request.body()
        return {"status": "ok"}

    fastapi_app.add_middleware(RequestLimitsMiddleware)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/query",
        "raw_path": b"/query",
        "query_string": b"",
        "headers": [],
        "client": ("192.0.2.1", 5000),
        "server": ("test", 80),
    }
    responses = []

    async def receive():
        return {"type": "http.request", "body": b"12345", "more_body": False}

    async def send(message):
        responses.append(message)

    asyncio.run(fastapi_app(scope, receive, send))
    assert responses[0]["status"] == 413

import json

from fastapi.routing import APIRoute

import vllm_mlx.server as server
from vllm_mlx.middleware import metrics
from vllm_mlx.middleware.metrics import MetricsMiddleware
from vllm_mlx.request_metrics import RequestRecorder
from vllm_mlx.routes.health import router as health_router


async def _empty_receive():
    return {"type": "http.request", "body": b"", "more_body": False}


def _sse_event(payload: dict, *, crlf: bool = False, terminate: bool = True) -> bytes:
    ending = b"\r\n\r\n" if crlf and terminate else b"\n\n" if terminate else b"\n"
    return b"data: " + json.dumps(payload).encode("utf-8") + ending


def test_request_metrics_middleware_is_not_enabled_by_default():
    assert all(
        middleware.cls is not MetricsMiddleware
        for middleware in server.app.user_middleware
    )


def test_requests_endpoint_requires_api_key_dependency():
    route = next(
        route
        for route in health_router.routes
        if isinstance(route, APIRoute) and route.path == "/v1/requests"
    )

    assert route.dependant.dependencies


async def test_metrics_middleware_drains_final_sse_event(monkeypatch):
    recorder = RequestRecorder()
    monkeypatch.setattr(metrics, "get_recorder", lambda: recorder)

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": _sse_event(
                    {"choices": [{"delta": {"content": "hello"}}]},
                    crlf=True,
                ),
                "more_body": True,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": _sse_event(
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                    },
                    terminate=False,
                ),
                "more_body": False,
            }
        )

    sent = []

    async def send(message):
        sent.append(message)

    middleware = MetricsMiddleware(app)
    await middleware(
        {"type": "http", "path": "/v1/chat/completions"},
        _empty_receive,
        send,
    )

    entries = recorder.entries()
    assert len(entries) == 1
    assert entries[0]["finish_reason"] == "stop"
    assert entries[0]["prompt_tokens"] == 2
    assert entries[0]["generated_tokens"] == 3
    assert entries[0]["message_preview"] == "hello"

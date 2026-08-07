import json

import httpx
import pytest
from fastapi import HTTPException

from llm_node import state
from llm_node.config import AppConfig, ModelConfig, ProgramConfig, Scheme
from llm_node.gateway import proxy
from llm_node.gateway.aliases import resolve_alias_checked
from llm_node.state import ModelStatus


def _cfg():
    m = ModelConfig(
        primary_name="m1",
        aliases=("m1", "alias1"),
        mode="Chat",
        port=8000,
        schemes={"s": Scheme("s", frozenset({"rtx 4060"}), "r.cmd", {"rtx 4060": 2048})},
    )
    return AppConfig(
        program=ProgramConfig(host="127.0.0.1", port=8080, alive_time=60, log_level="INFO"),
        models={"m1": m},
    )


# ---------- helpers ----------
def test_strip_headers_removes_hop_by_hop():
    out = proxy._strip_headers(
        {
            "host": "x",
            "content-length": "3",
            "transfer-encoding": "chunked",
            "authorization": "Bearer t",
        },
        extra=("host",),
    )
    assert "host" not in out and "content-length" not in out and "transfer-encoding" not in out
    assert out["authorization"] == "Bearer t"


def test_strip_headers_response_side_drops_length_encoding():
    out = proxy._strip_headers(
        {
            "content-length": "9",
            "content-encoding": "gzip",
            "transfer-encoding": "chunked",
            "connection": "keep-alive",
            "content-type": "application/json",
        },
        extra=("connection", "content-encoding"),
    )
    for bad in ("content-length", "content-encoding", "transfer-encoding", "connection"):
        assert bad not in out
    assert out["content-type"] == "application/json"


def test_detect_sse_by_content_type():
    class R:
        headers = {"content-type": "text/event-stream"}  # noqa: RUF012

    assert proxy._detect_sse(R()) is True

    class R2:
        headers = {"content-type": "application/json"}  # noqa: RUF012

    assert proxy._detect_sse(R2()) is False


def test_extract_model_alias_from_dict():
    assert proxy._extract_model_alias({"model": "m1"}) == "m1"
    assert proxy._extract_model_alias(b"raw") is None
    assert proxy._extract_model_alias({}) is None


def test_is_stream_flag():
    assert proxy._is_stream({"stream": True}) is True
    assert proxy._is_stream({"stream": False}) is False
    assert proxy._is_stream({}) is False
    assert proxy._is_stream(b"raw") is False


def test_reserialize_roundtrip():
    body = {"model": "m1", "stream": True}
    assert json.loads(proxy._reserialize(body)) == body


# ---------- resolve_alias_checked / _get_or_create_client ----------
def test_resolve_alias_unknown_raises_404():
    with pytest.raises(HTTPException) as ei:
        resolve_alias_checked(_cfg(), "nope")
    assert ei.value.status_code == 404


def test_resolve_alias_missing_model_raises_400():
    with pytest.raises(HTTPException) as ei:
        resolve_alias_checked(_cfg(), None)
    assert ei.value.status_code == 400


def test_resolve_alias_normal():
    assert resolve_alias_checked(_cfg(), "alias1") == "m1"


def test_get_or_create_client_lazy_and_reuse():
    import asyncio

    pool: dict = {}
    c1 = proxy._get_or_create_client(pool, 8000)
    c2 = proxy._get_or_create_client(pool, 8000)
    assert c1 is c2 and 8000 in pool
    asyncio.run(c1.aclose())


# ---------- _stream_wrapper ----------
async def test_stream_wrapper_forwards_chunks_ends_request():
    state._reset()
    state.set_status("m1", ModelStatus.ROUTING, force=True)
    state.begin_request("m1")

    class FakeResp:
        headers = {"content-type": "text/event-stream"}  # noqa: RUF012

        async def aiter_bytes(self):
            for c in [b'data: {"choices":[]}\n\n', b'data: {"x":1}\n\n']:
                yield c

        async def aclose(self):
            pass

    out = [c async for c in proxy._stream_wrapper(FakeResp(), "m1")]
    assert len(out) == 2
    assert state.pending_count("m1") == 0


# ---------- forward ----------
def _make_request(method, path, json_body, content_type="application/json"):
    from starlette.requests import Request as StarletteRequest

    body = json.dumps(json_body).encode() if json_body is not None else b""
    scope = {
        "type": "http",
        "method": method,
        "path": path.split("/"),
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", content_type.encode()), (b"host", b"x")]
        + ([(b"content-length", str(len(body)).encode())] if body else []),
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return StarletteRequest(scope, receive)


class FakeLifecycle:
    def __init__(self, status=None):
        self._status = status if status is not None else ModelStatus.ROUTING

    async def ensure_running(self, alias, *, inc_pending=False):
        if inc_pending and self._status == ModelStatus.ROUTING:
            state.begin_request(alias)
        return self._status


async def test_forward_non_stream_passes_through_and_ends_request():
    state._reset()

    def handler(req):
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
            },
            headers={"content-type": "application/json"},
        )

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(handler)
    )
    req = _make_request("POST", "v1/chat/completions", {"model": "m1", "stream": False})
    resp = await proxy.forward(req, "v1/chat/completions", FakeLifecycle(), _cfg(), {8000: client})
    assert resp.status_code == 200
    assert state.pending_count("m1") == 0
    await client.aclose()


async def test_forward_stream_returns_streaming_and_ends_on_consume():
    state._reset()
    sse = b'data: {"usage":{"prompt_tokens":2,"completion_tokens":3}}\n\n'

    def handler(req):
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(handler)
    )
    req = _make_request("POST", "v1/chat/completions", {"model": "m1", "stream": True})
    resp = await proxy.forward(req, "v1/chat/completions", FakeLifecycle(), _cfg(), {8000: client})
    assert resp.status_code == 200
    consumed = b"".join([chunk async for chunk in resp.body_iterator])
    assert b"usage" in consumed
    assert state.pending_count("m1") == 0
    await client.aclose()


async def test_forward_ensure_running_failed_returns_503():
    state._reset()
    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    )
    req = _make_request("POST", "v1/chat/completions", {"model": "m1"})
    with pytest.raises(HTTPException) as ei:
        await proxy.forward(
            req,
            "v1/chat/completions",
            FakeLifecycle(ModelStatus.FAILED),
            _cfg(),
            {8000: client},
        )
    assert ei.value.status_code == 503
    assert state.pending_count("m1") == 0
    await client.aclose()


async def test_forward_alias_unknown_returns_404():
    state._reset()
    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    )
    req = _make_request("POST", "v1/chat/completions", {"model": "nope"})
    with pytest.raises(HTTPException) as ei:
        await proxy.forward(req, "v1/chat/completions", FakeLifecycle(), _cfg(), {8000: client})
    assert ei.value.status_code == 404
    await client.aclose()


async def test_forward_missing_model_returns_400():
    state._reset()
    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    )
    req = _make_request("POST", "v1/chat/completions", {"messages": []})
    with pytest.raises(HTTPException) as ei:
        await proxy.forward(req, "v1/chat/completions", FakeLifecycle(), _cfg(), {8000: client})
    assert ei.value.status_code == 400
    assert state.pending_count("m1") == 0
    await client.aclose()


async def test_forward_upstream_5xx_passes_through_raw():
    state._reset()
    raw = b'{"error":{"message":"model overloaded","type":"server_error"}}'

    def handler(req):
        return httpx.Response(503, content=raw, headers={"content-type": "application/json"})

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(handler)
    )
    req = _make_request("POST", "v1/chat/completions", {"model": "m1"})
    resp = await proxy.forward(req, "v1/chat/completions", FakeLifecycle(), _cfg(), {8000: client})
    assert resp.status_code == 503
    assert resp.body == raw
    assert b'"detail"' not in resp.body
    assert state.pending_count("m1") == 0
    await client.aclose()

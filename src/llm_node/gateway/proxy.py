"""Reverse proxy: alias resolve → lifecycle ensure_running → httpx forward →
SSE/non-SSE branch → end_request. Ported from LLM-Manager, stripped of token
metering/DB recording (stateless node).

end_request 分布三处(非 stream return 前 / 各 except / _stream_wrapper finally)
防 pending 泄漏。_strip_headers(extra=connection/content-encoding) 去 hop-by-hop 头。"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from llm_node.gateway.aliases import resolve_alias_checked

logger = logging.getLogger(__name__)

_STRIP_BASE = {"content-length", "transfer-encoding"}


def _strip_headers(headers: Mapping[str, str], extra: tuple[str, ...] = ()) -> dict[str, str]:
    bad = _STRIP_BASE | set(extra)
    return {k: v for k, v in headers.items() if k.lower() not in bad}


def _detect_sse(resp) -> bool:
    return "text/event-stream" in resp.headers.get("content-type", "")


def _extract_model_alias(body) -> str | None:
    return body.get("model") if isinstance(body, dict) else None


def _is_stream(body) -> bool:
    return isinstance(body, dict) and body.get("stream") is True


def _reserialize(body: dict) -> bytes:
    return json.dumps(body).encode("utf-8")


async def _read_body(request: Request):
    if "application/json" in request.headers.get("content-type", ""):
        try:
            return await request.json()
        except Exception:  # noqa: BLE001
            return await request.body()
    return await request.body()


def _get_or_create_client(pool: dict, port: int) -> httpx.AsyncClient:
    client = pool.get(port)
    if client is None:
        client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            timeout=httpx.Timeout(30.0, read=600.0, connect=30.0, write=30.0),
        )
        pool[port] = client
    return client


async def _stream_wrapper(resp, model):
    from llm_node import state

    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
    finally:
        await resp.aclose()
        state.end_request(model)


async def forward(request: Request, path: str, lifecycle, cfg, client_pool) -> Response:
    from llm_node import state
    from llm_node.state import ModelStatus

    t0 = time.monotonic()
    body = await _read_body(request)
    alias = _extract_model_alias(body)
    primary = resolve_alias_checked(cfg, alias)
    logger.info("REQ %s /%s model=%s", request.method, path, primary)
    served = cfg.models[primary].aliases[0]  # aliases[0]=主别名=下游 served name
    if isinstance(body, dict):
        body["model"] = served
        request_data = _reserialize(body)
    else:
        request_data = body if isinstance(body, bytes) else b""

    status = await lifecycle.ensure_running(primary, inc_pending=True)
    if status != ModelStatus.ROUTING:
        logger.warning("model %s not routing (%s)", primary, status.value)
        raise HTTPException(503, f"model '{primary}' not routing (status={status.value})")

    try:
        port = cfg.models[primary].port
        client = _get_or_create_client(client_pool, port)
        resp = await client.send(
            client.build_request(
                request.method,
                path,
                headers=_strip_headers(request.headers, extra=("host",)),
                content=request_data,
                params=request.query_params,
            ),
            stream=True,
        )
        if _detect_sse(resp):
            logger.info(
                "RESP %d stream model=%s %.2fs", resp.status_code, primary, time.monotonic() - t0
            )
            return StreamingResponse(
                _stream_wrapper(resp, primary),
                status_code=resp.status_code,
                headers=_strip_headers(resp.headers, extra=("connection", "content-encoding")),
            )
        content = await resp.aread()
        await resp.aclose()
        state.end_request(primary)
        logger.info("RESP %d model=%s %.2fs", resp.status_code, primary, time.monotonic() - t0)
        return Response(
            content=content,
            status_code=resp.status_code,
            headers=_strip_headers(resp.headers, extra=("connection", "content-encoding")),
        )
    except HTTPException:
        state.end_request(primary)
        raise
    except httpx.HTTPError as e:
        state.end_request(primary)
        logger.warning("upstream error model=%s: %s", primary, e)
        raise HTTPException(502, f"upstream error: {e}")
    except Exception as e:  # noqa: BLE001
        state.end_request(primary)
        logger.warning("internal model=%s: %s", primary, e)
        raise HTTPException(500, f"internal: {e}")


def register_proxy_routes(
    app: FastAPI,
    lifecycle,
    client_pool,
) -> None:
    """挂载 OpenAI 兼容代理的 catch-all(POST/PUT/DELETE/PATCH)。读穿:每请求取 fresh cfg。"""

    async def _forward(path: str, request: Request) -> Response:
        cfg = request.app.state.config_store.snapshot()
        return await forward(request, path, lifecycle, cfg, client_pool)

    @app.post("/{path:path}", operation_id="catch_all__path__post")
    async def catch_all_post(path: str, request: Request) -> Response:
        return await _forward(path, request)

    @app.put("/{path:path}", operation_id="catch_all__path__put")
    async def catch_all_put(path: str, request: Request) -> Response:
        return await _forward(path, request)

    @app.delete("/{path:path}", operation_id="catch_all__path__delete")
    async def catch_all_delete(path: str, request: Request) -> Response:
        return await _forward(path, request)

    @app.patch("/{path:path}", operation_id="catch_all__path__patch")
    async def catch_all_patch(path: str, request: Request) -> Response:
        return await _forward(path, request)

"""Health probes by model mode. 2-phase (shallow /v1/models + deep per-mode),
shared start_time/timeout budget. httpx (no openai SDK). Never raises:未知 mode
返回失败结果(不抛),所有路径产出 ProbeResult。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx


@dataclass(frozen=True, slots=True)
class ProbeResult:
    ok: bool
    message: str


def _make_client(port: int) -> httpx.Client:
    return httpx.Client(base_url=f"http://127.0.0.1:{port}/v1", timeout=5.0)


def _deep_request(mode: str) -> tuple[str, dict] | None:
    """Pure: (path, json_body_template_without_model) relative to base /v1。"""
    if mode == "Chat":
        return "/chat/completions", {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "stream": False,
        }
    if mode == "Embedding":
        return "/embeddings", {"input": "hello", "encoding_format": "float"}
    if mode == "Reranker":
        return "/rerank", {
            "query": "hello",
            "documents": ["hello world", "test document"],
            "top_n": 1,
        }
    return None


def _probe(
    mode: str, label: str, alias: str, port: int, start_time: float | None, timeout: float
) -> ProbeResult:
    if start_time is None:
        start_time = time.monotonic()
    client = _make_client(port)
    try:
        ok = False
        while time.monotonic() - start_time < timeout:
            try:
                if client.get("/models", timeout=3.0).status_code < 400:
                    ok = True
                    break
            except Exception:  # noqa: BLE001, S110
                pass
            time.sleep(2)
        if not ok:
            return ProbeResult(False, f"{label}探测器浅层检查超时: 服务在 {timeout:.0f} 秒内不可用")
        deep = _deep_request(mode)
        if deep is None:
            return ProbeResult(False, f"{label}探测器不支持的模式: {mode}")
        path, body = deep
        while time.monotonic() - start_time < timeout:
            try:
                resp = client.post(path, json={**body, "model": alias}, timeout=5.0)
                if resp.status_code < 400:
                    return ProbeResult(True, f"{label}探测器健康检查成功")
            except Exception:  # noqa: BLE001, S110
                pass
            time.sleep(1)
        return ProbeResult(False, f"{label}探测器深层检查超时")
    finally:
        client.close()


def probe_chat(alias, port, start_time=None, timeout=300) -> ProbeResult:
    return _probe("Chat", "聊天", alias, port, start_time, timeout)


def probe_embedding(alias, port, start_time=None, timeout=300) -> ProbeResult:
    return _probe("Embedding", "嵌入", alias, port, start_time, timeout)


def probe_reranker(alias, port, start_time=None, timeout=300) -> ProbeResult:
    return _probe("Reranker", "重排序", alias, port, start_time, timeout)


probe_registry: dict[str, Callable[..., ProbeResult]] = {
    "Chat": probe_chat,
    "Embedding": probe_embedding,
    "Reranker": probe_reranker,
}

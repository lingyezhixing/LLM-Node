import httpx

from llm_node import probes
from llm_node.probes import ProbeResult, _deep_request, probe_registry


def test_registry_has_all_three_modes():
    assert set(probe_registry) == {"Chat", "Embedding", "Reranker"}


def test_deep_request_shape_per_mode():
    assert _deep_request("Chat")[0] == "/chat/completions"
    assert "messages" in _deep_request("Chat")[1]
    assert _deep_request("Embedding")[0] == "/embeddings"
    assert _deep_request("Reranker")[0] == "/rerank"


def test_probe_chat_success_via_mock_transport(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={})

    client = httpx.Client(
        base_url="http://127.0.0.1:9999/v1", transport=httpx.MockTransport(handler)
    )
    monkeypatch.setattr(probes, "_make_client", lambda port: client)
    result = probes.probe_chat("alias", 9999, timeout=10.0)
    assert isinstance(result, ProbeResult)
    assert result.ok is True
    client.close()


def test_probe_returns_failure_when_shallow_never_succeeds(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = httpx.Client(
        base_url="http://127.0.0.1:9999/v1", transport=httpx.MockTransport(handler)
    )
    monkeypatch.setattr(probes, "_make_client", lambda port: client)
    result = probes.probe_chat("alias", 9999, timeout=0.5)
    assert result.ok is False
    client.close()

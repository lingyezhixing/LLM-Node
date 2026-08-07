from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_node import config
from llm_node.config_store import ConfigStore
from llm_node.gateway.routes import register_routes
from llm_node.state import ModelStatus

_CFG_BODY = """
program: {host: 0.0.0.0, port: 8080, alive_time: 60, log_level: INFO}
Local-Models:
  m1:
    aliases: ["m1"]
    mode: Chat
    port: 8000
    RTX4060:
      required_devices: ["rtx 4060"]
      script_path: "q.bat"
      memory_mb: {"rtx 4060": 2048}
"""

_CFG_DISTINCT = """
program: {host: 0.0.0.0, port: 8080, alive_time: 60, log_level: INFO}
Local-Models:
  internal-qwen-key:
    aliases: ["qwen2.5-32b-instruct"]
    mode: Chat
    port: 8001
    RTX4060:
      required_devices: ["rtx 4060"]
      script_path: "q.bat"
      memory_mb: {"rtx 4060": 2048}
"""


def _cfg(tmp_path: Path) -> config.AppConfig:
    p = tmp_path / "config.yaml"
    p.write_text(_CFG_BODY, encoding="utf-8")
    return config.load(p)


class _FakeLife:
    async def ensure_running(self, alias, *, inc_pending=False):
        return ModelStatus.ROUTING


def _register(app, cfg, client_pool=None):
    store = ConfigStore.__new__(ConfigStore)
    store._snapshot = cfg
    store._path = None
    register_routes(app, _FakeLife(), client_pool or {})
    app.state.config_store = store


def test_health_returns_200(tmp_path):
    app = FastAPI()
    _register(app, _cfg(tmp_path))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200


def test_v1_models_returns_catalog(tmp_path):
    app = FastAPI()
    _register(app, _cfg(tmp_path))
    with TestClient(app) as c:
        r = c.get("/v1/models")
    assert r.status_code == 200 and "m1" in {m["id"] for m in r.json()["data"]}


def test_v1_models_lists_primary_alias_not_internal_key(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(_CFG_DISTINCT, encoding="utf-8")
    cfg = config.load(p)
    app = FastAPI()
    _register(app, cfg)
    with TestClient(app) as c:
        r = c.get("/v1/models")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert "qwen2.5-32b-instruct" in ids
    assert "internal-qwen-key" not in ids


def test_options_preflight_returns_204_with_cors(tmp_path):
    app = FastAPI()
    _register(app, _cfg(tmp_path))
    with TestClient(app) as c:
        r = c.options("/v1/chat/completions")
    assert r.status_code == 204 and r.headers.get("access-control-allow-origin") == "*"


def test_non_get_catchall_forwards_to_proxy(tmp_path):
    def fail_handler(req):
        raise httpx.ConnectError("no upstream (test)", request=req)

    app = FastAPI()
    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(fail_handler)
    )
    _register(app, _cfg(tmp_path), client_pool={8000: client})
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={"model": "m1"})
    assert r.status_code == 502

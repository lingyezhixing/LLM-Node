import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_node import config, state
from llm_node.config_store import ConfigStore
from llm_node.gateway.api.models import build_models_response
from llm_node.gateway.routes import register_routes
from llm_node.state import ModelStatus

_CFG = """
program: {host: 0.0.0.0, port: 8080, alive_time: 60, log_level: INFO}
Local-Models:
  internal-qwen-key:
    aliases: ["qwen2.5-32b"]
    mode: Chat
    port: 8001
    RTX4060:
      required_devices: ["rtx 4060"]
      script_path: "q.bat"
      memory_mb: {"rtx 4060": 2048}
"""


class _FakeLife:
    def __init__(self):
        self.started: list[str] = []
        self.stopped: list[str] = []

    async def ensure_running(self, alias, *, inc_pending=False):
        self.started.append(alias)
        return ModelStatus.ROUTING

    async def stop(self, alias):
        self.stopped.append(alias)
        return state.get_status(alias)

    async def unload_all(self):
        return []


def _cfg(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(_CFG, encoding="utf-8")
    return config.load(p)


def _app(tmp_path, life=None):
    life = _FakeLife() if life is None else life
    cfg = _cfg(tmp_path)
    store = ConfigStore.__new__(ConfigStore)
    store._snapshot = cfg
    store._path = None
    app = FastAPI()
    register_routes(app, life, {})
    app.state.config_store = store
    return app


def test_list_models_snapshot(tmp_path):
    state._reset()
    cfg = _cfg(tmp_path)
    resp = build_models_response(cfg)
    assert resp.data[0].alias == "qwen2.5-32b"
    assert resp.data[0].status == "stopped"
    state._reset()


def test_info_unknown_alias_404(tmp_path):
    state._reset()
    app = _app(tmp_path)
    with TestClient(app) as c:
        r = c.get("/api/models/nope/info")
    assert r.status_code == 404
    state._reset()


def test_info_returns_status(tmp_path):
    state._reset()
    app = _app(tmp_path)
    state.set_status("internal-qwen-key", ModelStatus.ROUTING, force=True)
    with TestClient(app) as c:
        r = c.get("/api/models/qwen2.5-32b/info")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "routing"
    assert body["alias"] == "qwen2.5-32b"
    assert body["mode"] == "Chat"
    state._reset()


def test_start_unknown_alias_404(tmp_path):
    state._reset()
    app = _app(tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/models/nope/start")
    assert r.status_code == 404
    state._reset()


def test_start_when_routing_409(tmp_path):
    state._reset()
    app = _app(tmp_path)
    state.set_status("internal-qwen-key", ModelStatus.ROUTING, force=True)
    with TestClient(app) as c:
        r = c.post("/api/models/qwen2.5-32b/start")
    assert r.status_code == 409
    state._reset()


def test_start_accepted_202_and_fires_ensure_running(tmp_path):
    state._reset()
    life = _FakeLife()
    app = _app(tmp_path, life)
    with TestClient(app) as c:
        r = c.post("/api/models/qwen2.5-32b/start")
        assert r.status_code == 202
        for _ in range(50):
            if life.started:
                break
            time.sleep(0.02)
    assert life.started == ["internal-qwen-key"]
    state._reset()


def test_stop_accepted_202_and_fires_stop(tmp_path):
    state._reset()
    life = _FakeLife()
    app = _app(tmp_path, life)
    with TestClient(app) as c:
        r = c.post("/api/models/qwen2.5-32b/stop")
        assert r.status_code == 202
        for _ in range(50):
            if life.stopped:
                break
            time.sleep(0.02)
    assert life.stopped == ["internal-qwen-key"]
    state._reset()


def test_stop_all_202(tmp_path):
    state._reset()
    app = _app(tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/models/stop-all")
    assert r.status_code == 202
    state._reset()


def test_restart_accepted_202_and_fires_stop_then_start(tmp_path):
    state._reset()
    life = _FakeLife()
    app = _app(tmp_path, life)
    with TestClient(app) as c:
        r = c.post("/api/models/qwen2.5-32b/restart")
        assert r.status_code == 202
        for _ in range(50):
            if life.stopped and life.started:
                break
            time.sleep(0.02)
    assert life.stopped == ["internal-qwen-key"]
    assert life.started == ["internal-qwen-key"]
    state._reset()


def test_restart_unknown_alias_404(tmp_path):
    state._reset()
    app = _app(tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/models/nope/restart")
    assert r.status_code == 404
    state._reset()

import pytest
from fastapi.testclient import TestClient

from llm_node.app import create_app

_CFG = """
program: {host: 0.0.0.0, port: 8080, alive_time: 60, log_level: INFO}
Local-Models:
  Qwen3-4B:
    aliases: ["Qwen3-4B"]
    mode: Chat
    port: 10001
    RTX4060:
      required_devices: ["rtx 4060"]
      script_path: "q.bat"
      memory_mb: {"rtx 4060": 5120}
"""


def test_app_boots_and_health_ok(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_CFG, encoding="utf-8")
    app = create_app(config_path=cfg_path)
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200


def test_create_app_validates_and_fails_fast_on_bad_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "program: {host: 0.0.0.0, port: 8080, alive_time: 60, log_level: INFO}\n"
        "Local-Models:\n"
        "  A: {aliases: [x], mode: Chat, port: 1}\n"
        "  B: {aliases: [x], mode: Chat, port: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        create_app(config_path=cfg_path)

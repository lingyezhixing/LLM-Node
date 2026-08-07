from fastapi.testclient import TestClient

from llm_node.app import create_app

_CFG_BODY = """
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


def test_lifespan_boots_monitor_and_cleans_up(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_CFG_BODY, encoding="utf-8")
    app = create_app(config_path=cfg_path)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    # shutdown 后无异常(unload_all + 关闭 clients 正常收口)

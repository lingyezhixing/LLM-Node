"""Test-wide fixtures.

create_app() 调用 setup_logging(),它每次启动挂一个时间戳文件
logs/llm-node_{ts}.log 到 root logger。不加隔离会把 pytest 输出混进生产日志文件。
Stub setup_logging 整个套件,测试永不触碰真实日志文件。
"""

import pytest

from llm_node import app


@pytest.fixture(autouse=True)
def _isolate_logging(monkeypatch):
    monkeypatch.setattr(app, "setup_logging", lambda *a, **k: None)

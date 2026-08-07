"""/api/* 子路由共享的每请求访问器。"""

from __future__ import annotations

from fastapi import Request

from llm_node.config import AppConfig
from llm_node.config_store import ConfigStore


def get_config_store(request: Request) -> ConfigStore:
    return request.app.state.config_store


def get_config(request: Request) -> AppConfig:
    return request.app.state.config_store.snapshot()

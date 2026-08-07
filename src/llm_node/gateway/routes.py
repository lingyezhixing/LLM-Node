"""Gateway HTTP layer composition root. Wires the management API (/api/*),
catalog (/health, /v1/models, OPTIONS preflight), and the OpenAI-compatible
proxy catch-all."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI

from llm_node.gateway import catalog, proxy
from llm_node.gateway.api.models import register_models_routes


def register_routes(app: FastAPI, lifecycle, client_pool) -> None:
    api = APIRouter(prefix="/api")
    register_models_routes(api, lifecycle)
    app.include_router(api)
    catalog.register_catalog(app)  # /health, /v1/models, OPTIONS 预检
    proxy.register_proxy_routes(app, lifecycle, client_pool)  # OpenAI 代理 catch-all

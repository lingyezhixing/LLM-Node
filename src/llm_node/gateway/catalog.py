"""Catalog + preflight routes: GET /health, GET /v1/models (id=aliases[0]),
OPTIONS preflight short-circuit (204 + CORS, before body/alias)."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

_CORS = {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,PUT,DELETE,PATCH,OPTIONS",
    "access-control-allow-headers": "Content-Type, Authorization",
}


def register_catalog(app: FastAPI) -> None:
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models(request: Request) -> dict:
        # id = aliases[0](主别名 = 下游 served name = 客户端调用名);primary_name 仅为内部键,不外露。
        # 读穿:每请求取 fresh 快照(reload 后新别名可路由)。
        cfg = request.app.state.config_store.snapshot()
        data = [{"id": m.aliases[0], "object": "model"} for m in cfg.models.values()]
        return {"object": "list", "data": data}

    @app.options("/{path:path}")
    def preflight(path: str) -> JSONResponse:
        return JSONResponse(status_code=204, content={}, headers=_CORS)

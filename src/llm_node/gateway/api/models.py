"""Model lifecycle management endpoints: info / start / stop / restart / stop-all.

Stateless node port of the manager's /api/models/* minus the SSE stream
(no frontend consumer). Pydantic schemas → OpenAPI."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from llm_node import state
from llm_node.config import AppConfig
from llm_node.gateway.aliases import resolve_alias_checked
from llm_node.gateway.api.common import get_config_store


class ModelInfo(BaseModel):
    alias: str  # cfg.aliases[0] — external identity (same as /v1/models)
    mode: str
    port: int
    auto_start: bool
    status: str  # state.ModelStatus value
    pid: int | None
    pending: int
    failure_reason: str | None
    started_at: float | None  # wall-clock epoch when entered ROUTING (None if not routing)
    last_access: float  # wall-clock epoch of last activity (0.0 if never)


class ModelsResponse(BaseModel):
    data: list[ModelInfo]


def build_models_response(cfg: AppConfig) -> ModelsResponse:
    """Current model snapshot from module-level state + cfg. Shared by list/info."""
    items: list[ModelInfo] = []
    for name, m in cfg.models.items():
        items.append(
            ModelInfo(
                alias=m.aliases[0],
                mode=m.mode,
                port=m.port,
                auto_start=m.auto_start,
                status=state.get_status(name).value,
                pid=state.get_pid(name),
                pending=state.pending_count(name),
                failure_reason=state.get_failure_reason(name),
                started_at=state.get_started_at(name),
                last_access=state.get_last_access_wall(name),
            )
        )
    return ModelsResponse(data=items)


def _find_primary(request: Request, alias: str) -> str:
    return resolve_alias_checked(get_config_store(request).snapshot(), alias)


async def _do_restart(lifecycle, primary: str) -> None:
    """restart = stop → ensure_running。lifecycle 读穿 → ensure_running 拿最新配置。"""
    await lifecycle.stop(primary)
    await lifecycle.ensure_running(primary)


def register_models_routes(router: APIRouter, lifecycle) -> None:
    @router.get("/models", response_model=ModelsResponse)
    async def list_models(request: Request) -> ModelsResponse:
        return build_models_response(get_config_store(request).snapshot())

    @router.get("/models/{alias}/info", response_model=ModelInfo)
    async def model_info(alias: str, request: Request) -> ModelInfo:
        primary = _find_primary(request, alias)
        cfg = get_config_store(request).snapshot()
        m = cfg.models[primary]
        return ModelInfo(
            alias=m.aliases[0],
            mode=m.mode,
            port=m.port,
            auto_start=m.auto_start,
            status=state.get_status(primary).value,
            pid=state.get_pid(primary),
            pending=state.pending_count(primary),
            failure_reason=state.get_failure_reason(primary),
            started_at=state.get_started_at(primary),
            last_access=state.get_last_access_wall(primary),
        )

    @router.post("/models/{alias}/start", status_code=202)
    async def start_model(alias: str, request: Request) -> Response:
        primary = _find_primary(request, alias)
        if state.is_runnable(primary):
            raise HTTPException(409, f"model '{alias}' already routing")
        asyncio.create_task(lifecycle.ensure_running(primary))
        return Response(status_code=202)

    @router.post("/models/{alias}/stop", status_code=202)
    async def stop_model(alias: str, request: Request) -> Response:
        primary = _find_primary(request, alias)
        asyncio.create_task(lifecycle.stop(primary))
        return Response(status_code=202)

    @router.post("/models/{alias}/restart", status_code=202)
    async def restart_model(alias: str, request: Request) -> Response:
        primary = _find_primary(request, alias)
        asyncio.create_task(_do_restart(lifecycle, primary))
        return Response(status_code=202)

    @router.post("/models/stop-all", status_code=202)
    async def stop_all_models(request: Request) -> Response:
        asyncio.create_task(lifecycle.unload_all())
        return Response(status_code=202)

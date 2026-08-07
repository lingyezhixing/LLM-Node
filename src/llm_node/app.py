"""Composition root: setup_logging + load/validate YAML config + FastAPI app with lifespan.

lifespan initializes the DeviceMonitor (initial refresh), the Supervisor, an
httpx-client pool; closes them on shutdown. Stateless: no DB, no tray, no SSE feeds.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI

from llm_node import config
from llm_node.config_store import ConfigStore
from llm_node.devices import DeviceMonitor, build_adapters
from llm_node.gateway.routes import register_routes
from llm_node.logging_setup import setup_logging
from llm_node.probes import probe_registry
from llm_node.runtime import background
from llm_node.runtime.lifecycle import Lifecycle
from llm_node.supervisor import Supervisor

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.environ.get("LLM_NODE_CONFIG", "config.yaml"))


def create_app(config_path: Path | None = None) -> FastAPI:
    resolved = Path(config_path or CONFIG_PATH)
    store = ConfigStore(resolved)
    cfg = store.snapshot()
    setup_logging(level=cfg.program.log_level)
    logger.info(
        "config loaded (%s): %d models, %s:%d, alive %dmin",
        resolved,
        len(cfg.models),
        cfg.program.host,
        cfg.program.port,
        cfg.program.alive_time,
    )
    # referenced 动态化:reload() 后重算设备引用,新模型引用的设备名进 online
    monitor = DeviceMonitor(build_adapters(), lambda: config.referenced_devices(store.snapshot()))
    supervisor = Supervisor()
    lifecycle = Lifecycle(
        get_cfg=store.snapshot,
        supervisor=supervisor,
        devices=monitor,
        probes=probe_registry,
        cwd=str(resolved.parent),
    )
    clients: dict[int, httpx.AsyncClient] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config_store = store
        app.state.monitor = monitor
        app.state.clients = clients
        app.state.lifecycle = lifecycle
        app.state.loop = asyncio.get_running_loop()
        await asyncio.to_thread(monitor.refresh)
        online = sorted(monitor.online_devices())
        logger.info("devices online: %s", ", ".join(online) if online else "(none)")

        stop_event = asyncio.Event()
        auto_models = config.auto_start_models(cfg)
        auto_task = asyncio.create_task(
            background.auto_start(
                lifecycle,
                auto_models,
                cfg,
                monitor,
                timeout=lifecycle.startup_timeout + background.AUTO_START_MARGIN,
                stop_event=stop_event,
            )
        )
        idle_task = asyncio.create_task(
            background.idle_reclamation_loop(lifecycle, store.snapshot, stop_event)
        )
        try:
            yield
        finally:
            stop_event.set()
            try:
                await lifecycle.unload_all()
            finally:
                if not idle_task.done():
                    idle_task.cancel()
                if not auto_task.done():
                    auto_task.cancel()
                await asyncio.gather(idle_task, auto_task, return_exceptions=True)
            for client in clients.values():
                await client.aclose()

    app = FastAPI(title="LLM-Node", lifespan=lifespan)
    register_routes(app, lifecycle, clients)
    return app


def create_dev_app() -> FastAPI:
    """No-arg factory for ``uvicorn --factory --reload`` (development mode)."""
    return create_app()


def run() -> None:
    """Direct entry: create app + uvicorn.run (blocking)."""
    import uvicorn

    app = create_app()
    cfg = app.state.config_store.snapshot()
    uvicorn.run(app, host=cfg.program.host, port=cfg.program.port, log_level="warning")

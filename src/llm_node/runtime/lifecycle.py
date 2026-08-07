"""Model lifecycle: start/stop coroutine pipeline + asyncio.Event cooperative
interruption + single-dispatch + crash->FAILED + reconcile safety net.

Ported from LLM-Manager v3, stripped of DB recording (no usage/runtime segments):
model stdout/stderr go to file sessions via ``model_log``. Spawn uses the scheme's
``script_path`` (stateless node approach) with variable substitution.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable

from llm_node import model_log as _logs
from llm_node import state
from llm_node.config import (
    AppConfig,
    ModelConfig,
    Scheme,
    resolve_alias,
    select_adaptive,
    substitute_vars,
)
from llm_node.probes import ProbeResult
from llm_node.runtime import scheduling
from llm_node.state import ModelStatus

logger = logging.getLogger(__name__)


class Lifecycle:
    def __init__(
        self,
        *,
        get_cfg: Callable[[], AppConfig],
        supervisor,
        devices,
        probes: dict[str, Callable],
        scheme_select=select_adaptive,
        startup_timeout: float = 60.0,
        cwd: str | None = None,
    ) -> None:
        self._get_cfg = get_cfg
        self._supervisor = supervisor
        self._devices = devices
        self._probes = probes
        self._scheme_select = scheme_select
        self.startup_timeout = startup_timeout
        self._cwd = cwd  # 启动脚本的工作目录(默认 None = 继承进程)
        self._stop_events: dict[str, asyncio.Event] = {}
        self._active_schemes: dict[str, Scheme] = {}
        self._spawn_lock = asyncio.Lock()  # 全局 spawn 锁:并发 spawn 串行,防显存超量
        self._log_session_ids: dict[str, int] = {}  # alias → 进行中模型日志会话 id

    # ---------- public ----------
    async def ensure_running(self, alias: str, *, inc_pending: bool = False) -> ModelStatus:
        self._reconcile(alias)
        if state.is_runnable(alias):
            status = state.get_status(alias)
            if inc_pending and status == ModelStatus.ROUTING:
                state.begin_request(alias)
            logger.debug("%s already %s (skip)", alias, status.value)
            return status
        future, won = state.claim_start(alias)
        if not won:
            try:
                await future
            except Exception:  # noqa: BLE001, S110
                pass
            status = state.get_status(alias)
            if inc_pending and status == ModelStatus.ROUTING:
                state.begin_request(alias)
            return status
        self._stop_events[alias] = asyncio.Event()
        try:
            status = await self._run_pipeline(alias)
            state.finish_start(alias, status, owner=future)
        except asyncio.CancelledError:
            state.record_failure(alias, "startup cancelled")
            state.finish_start(alias, ModelStatus.FAILED, owner=future)
            raise
        except Exception as e:  # noqa: BLE001
            state.record_failure(alias, f"pipeline error: {e}")
            state.finish_start(alias, ModelStatus.FAILED, owner=future)
        status = state.get_status(alias)
        if inc_pending and status == ModelStatus.ROUTING:
            state.begin_request(alias)
        return status

    async def stop(self, alias: str) -> ModelStatus:
        if state.get_status(alias) in (ModelStatus.STOPPED, ModelStatus.FAILED):
            return state.get_status(alias)
        state.set_status(alias, ModelStatus.STOPPED, force=True, reason="user stop")
        self._stop_events.setdefault(alias, asyncio.Event()).set()
        pid = state.get_pid(alias)
        if pid is not None:
            await self._supervisor.kill_tree(pid)
        state.clear_pid(alias)
        self._active_schemes.pop(alias, None)
        fut = state.pop_inflight(alias)
        if fut is not None and not fut.done():
            fut.set_result(ModelStatus.STOPPED)
        self._log_end(alias)  # 收口模型日志会话(落盘收尾):下次 start 起新会话
        return state.get_status(alias)

    async def unload_all(self) -> list[str]:
        cfg = self._get_cfg()
        names = [
            n
            for n in cfg.models
            if state.get_status(n) not in (ModelStatus.STOPPED, ModelStatus.FAILED)
        ]
        results = await asyncio.gather(*[self.stop(n) for n in names], return_exceptions=True)
        return [n for n, r in zip(names, results) if not isinstance(r, Exception)]

    # ---------- log session helpers ----------
    def _log_end(self, alias: str) -> None:
        sid = self._log_session_ids.pop(alias, None)
        if sid is not None:
            _logs.end_session(sid)

    # ---------- pipeline ----------
    async def _run_pipeline(self, alias: str) -> ModelStatus:
        ev = self._stop_events[alias]
        model = self._cfg_model(alias)

        await asyncio.to_thread(self._devices.refresh)
        if ev.is_set():
            return ModelStatus.STOPPED

        online = self._devices.online_devices()
        scheme = self._scheme_select(model, online)
        if scheme is None:
            required = sorted({d for s in model.schemes.values() for d in s.required_devices})
            msg = f"no adaptive scheme (required {required}, online {sorted(online)})"
            logger.warning("%s: %s", alias, msg)
            state.record_failure(alias, msg)
            return ModelStatus.FAILED
        logger.info(
            "cold start %s scheme=%s devices=%s",
            alias,
            scheme.config_source,
            sorted(scheme.required_devices),
        )

        # === spawn 锁:check_and_free + spawn 串行,避免并发 spawn 显存超量 ===
        async with self._spawn_lock:
            snap = self._devices.snapshot()
            runnable = self._runnable(exclude=alias)
            to_stop = scheduling.check_and_free(scheme.memory_mb, snap, runnable, time.monotonic())
            if to_stop:
                logger.info("evict %s to free mem for %s", list(to_stop), alias)
                await asyncio.gather(*[self.stop(n) for n in to_stop], return_exceptions=True)
                await asyncio.to_thread(self._devices.refresh)
                snap = self._devices.snapshot()
            if not self._deficit_satisfied(scheme.memory_mb, snap):
                logger.warning("%s: insufficient resource after eviction", alias)
                state.record_failure(alias, "insufficient resource after eviction")
                return ModelStatus.FAILED
            if ev.is_set():
                return ModelStatus.STOPPED

            # 脚本路径变量替换({{port}}/{{alias}});无占位符原样。
            script = substitute_vars(scheme.script_path, model)
            if os.name == "nt":
                # cmd /c 不认正斜杠路径('Model_startup_script/x.bat' 会解析成命令失败),
                # 归一化为反斜杠使批处理可被找到;POSIX 不做处理.
                script = os.path.normpath(script)
            env = {**os.environ}
            rec = await self._supervisor.spawn(
                script,
                shell=True,  # 脚本文件(.bat/.sh)需 shell 解释执行(原节点行为)
                env=env,
                cwd=self._cwd,
                on_output=lambda line, stream: _logs.capture(alias, line, stream),
            )
            logger.info("spawn %s pid=%d", alias, rec.pid)

            # === 模型日志会话:先收口上一会话(防快速 restart 残留),再开新会话。
            # 失败仅降级(该模型本次日志不落盘),不阻断 spawn:spawn 锁内不得抛。===
            try:
                self._log_end(alias)
                self._log_session_ids[alias] = _logs.start_session(
                    "model", model_name=alias, alias=model.aliases[0]
                )
            except Exception:
                logger.warning("log session start failed for %s", alias, exc_info=True)

            # === post-spawn 无-await 临界段 ===
            state.record_pid(alias, rec.pid)
            orphan_pid = rec.pid if ev.is_set() else None
            # === end critical section ===
        # === 锁外:orphan kill + probe 并行 ===
        if orphan_pid is not None:
            await self._supervisor.kill_tree(orphan_pid)
            self._log_end(
                alias
            )  # stop 在 spawn await 中到达(会话于其 _log_end 之后才开)→ 必须在此收口
            return ModelStatus.STOPPED

        # Any raise below must kill the spawned pid before propagating.
        try:
            if ev.is_set():
                return await self._abort_spawned(rec.pid)
            state.set_status(alias, ModelStatus.INIT_SCRIPT)
            state.set_status(alias, ModelStatus.HEALTH_CHECK)
            self._active_schemes[alias] = scheme

            if ev.is_set():
                return await self._abort_spawned(rec.pid)
            probe = await asyncio.to_thread(self._probe, alias, model.mode)
            logger.info("probe %s %s", alias, "ok" if probe.ok else "fail: " + str(probe.message))
            if ev.is_set():
                return await self._abort_spawned(rec.pid)
            if not probe.ok:
                await self._supervisor.kill_tree(rec.pid)
                if ev.is_set():
                    return ModelStatus.STOPPED
                self._log_end(alias)  # probe 失败不会走 on_exit / stop(FAILED 早退)→ 必须在此收口
                state.record_failure(alias, f"probe failed: {probe.message}")
                return ModelStatus.FAILED

            # === set-ROUTING 无-await 临界段 ===
            if ev.is_set():
                return await self._abort_spawned(rec.pid)
            state.set_status(alias, ModelStatus.ROUTING)
            state.touch_activity(alias)
            self._supervisor.on_exit(rec.pid, lambda code: self._on_crash(alias, code))
            logger.info("%s -> routing", alias)
            return ModelStatus.ROUTING
        except (Exception, asyncio.CancelledError):
            await self._supervisor.kill_tree(rec.pid)
            self._log_end(alias)
            raise

    async def _abort_spawned(self, pid: int | None) -> ModelStatus:
        if pid is not None:
            await self._supervisor.kill_tree(pid)
        return ModelStatus.STOPPED

    # ---------- crash / reconcile ----------
    def _on_crash(self, alias: str, code: int) -> None:
        try:
            if state.get_status(alias) == ModelStatus.STOPPED:
                return
            self._log_end(alias)
            state.record_failure(alias, f"process exited code={code}")
        except Exception as e:  # noqa: BLE001
            logger.error("on_exit callback error for %s: %s", alias, e)

    def _reconcile(self, alias: str) -> None:
        s = state.get_status(alias)
        if s in (ModelStatus.STOPPED, ModelStatus.FAILED):
            return
        pid = state.get_pid(alias)
        alive = pid is not None and self._supervisor.alive(pid)
        if s == ModelStatus.ROUTING and not alive:
            self._log_end(alias)
            state.record_failure(alias, f"reconcile: process dead (pid={pid})")
        elif s in (
            ModelStatus.STARTING,
            ModelStatus.INIT_SCRIPT,
            ModelStatus.HEALTH_CHECK,
        ) and not state.has_inflight(alias):
            state.record_failure(alias, f"reconcile: orphan {s.name} (no inflight)")
            state.clear_pid(alias)
            state.clear_inflight(alias)

    # ---------- helpers ----------
    def _deficit_satisfied(self, required: dict[str, int], snap: dict) -> bool:
        avail = {dev: info.available_memory_mb for dev, info in snap.items()}
        return not scheduling.compute_deficit(required, avail)

    def _cfg_model(self, alias: str) -> ModelConfig:
        cfg = self._get_cfg()
        return cfg.models[resolve_alias(cfg, alias)]

    def _runnable(self, exclude: str) -> dict[str, scheduling.RunnableInfo]:
        cfg = self._get_cfg()
        out: dict[str, scheduling.RunnableInfo] = {}
        for name in cfg.models:
            if name == exclude or not state.is_runnable(name):
                continue
            scheme = self._active_schemes.get(name)
            out[name] = scheduling.RunnableInfo(
                mem_mb=dict(scheme.memory_mb) if scheme else {},
                pending=state.pending_count(name),
                last_access=state.get_last_access(name),
            )
        return out

    def _probe(self, alias: str, mode: str) -> ProbeResult:
        model = self._cfg_model(alias)
        served = model.aliases[0]
        fn = self._probes[mode]
        return fn(served, model.port, None, self.startup_timeout)

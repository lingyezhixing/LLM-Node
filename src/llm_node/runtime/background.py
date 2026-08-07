"""Background loops: idle reclamation + auto-start."""

from __future__ import annotations

import asyncio
import logging
import time

from llm_node import state

logger = logging.getLogger(__name__)

AUTO_START_MARGIN: float = 30.0


def _plan_batches(models_schemes: list) -> tuple[list[str], list[str]]:
    """设备隔离贪心:scheme.required_devices 与并行批已占无交集 → parallel(累计 occupied);
    有交集 → serial。纯函数,无 IO。"""
    parallel: list[str] = []
    occupied: set[str] = set()
    serial: list[str] = []
    for name, scheme in models_schemes:
        if scheme.required_devices & occupied:
            serial.append(name)
        else:
            parallel.append(name)
            occupied |= scheme.required_devices
    return parallel, serial


def select_idle_candidates(alive_sec: float, now: float) -> list[str]:
    """只读 state:ROUTING ∩ pending==0 ∩ idle>alive_sec。now 注入(可测)。"""
    return [
        n
        for n in state.routing_names()
        if state.pending_count(n) == 0 and (now - state.get_last_access(n)) > alive_sec
    ]


async def idle_reclamation_loop(
    lifecycle, get_cfg, stop_event: asyncio.Event, *, period: float = 30.0
) -> None:
    """每轮从 get_cfg() 取 fresh alive_time。alive_time<=0 禁用。"""
    while not stop_event.is_set():
        try:
            alive_sec = get_cfg().program.alive_time * 60.0
            if alive_sec <= 0:
                logger.info("idle reclamation disabled (alive_time<=0)")
            else:
                now = time.monotonic()
                for name in select_idle_candidates(alive_sec, now):
                    if state.pending_count(name) > 0:
                        logger.info("skip reclaim %s: new request in flight", name)
                        continue
                    logger.info(
                        "idle reclaim %s (idle %.0fs)", name, now - state.get_last_access(name)
                    )
                    try:
                        await lifecycle.stop(name)
                    except Exception as e:  # noqa: BLE001
                        logger.error("idle reclaim stop failed %s: %s", name, e)
        except Exception as e:  # noqa: BLE001
            logger.error("idle reclamation iteration error: %s", e)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=period)
        except TimeoutError:
            pass


async def auto_start(
    lifecycle, models: list[str], cfg, monitor, *, timeout: float, stop_event: asyncio.Event
) -> None:
    """设备隔离分批调度:扫描硬件 → select_adaptive → _plan_batches
    → parallel gather(spawn 锁串行 spawn,probe 并行)+ serial 逐一(refresh 缓存刷新)。"""
    if not models:
        logger.info("no auto_start models")
        return
    logger.info("auto_start %d models: %s", len(models), models)
    from llm_node import config as _cfg

    async def _one(name: str) -> None:
        if stop_event.is_set():
            return
        try:
            status = await asyncio.wait_for(lifecycle.ensure_running(name), timeout)
            logger.info("auto_start %s -> %s", name, status.value)
        except TimeoutError:
            logger.warning("auto_start %s timeout (%.0fs)", name, timeout)
        except Exception as e:  # noqa: BLE001
            logger.error("auto_start %s failed: %s", name, e)

    await asyncio.to_thread(monitor.refresh)
    online = monitor.online_devices()
    planned = []
    for name in models:
        scheme = _cfg.select_adaptive(cfg.models[name], online)
        if scheme is None:
            required = sorted(
                {d for s in cfg.models[name].schemes.values() for d in s.required_devices}
            )
            logger.info(
                "auto_start skip %s: no adaptive scheme (required %s, online %s)",
                name,
                required,
                sorted(online),
            )
        else:
            planned.append((name, scheme))
    parallel, serial = _plan_batches(planned)
    logger.info("auto_start parallel=%s serial=%s", parallel, serial)
    if parallel:
        await asyncio.gather(*[_one(n) for n in parallel], return_exceptions=True)
    for name in serial:
        if stop_event.is_set():
            break
        await asyncio.to_thread(monitor.refresh)
        await _one(name)
    logger.info("auto_start batch complete")

import asyncio
import logging
import time

import pytest

from llm_node import state
from llm_node.config import AppConfig, ProgramConfig
from llm_node.runtime import background
from llm_node.state import ModelStatus


class _FakeLife:
    def __init__(self, stop=None, ensure_running=None):
        self.stopped: list[str] = []
        self.started: list[str] = []
        self._stop_fn = stop
        self._ensure_fn = ensure_running

    async def stop(self, name):
        state.set_status(name, ModelStatus.STOPPED, force=True)
        self.stopped.append(name)
        if self._stop_fn is not None:
            r = self._stop_fn(name)
            if asyncio.iscoroutine(r):
                await r

    async def ensure_running(self, name):
        self.started.append(name)
        if self._ensure_fn is not None:
            r = self._ensure_fn(name)
            if asyncio.iscoroutine(r):
                r = await r
            return r
        return ModelStatus.ROUTING


def _alive_cfg(alive: int) -> AppConfig:
    return AppConfig(
        program=ProgramConfig(host="x", port=1, alive_time=alive, log_level="INFO"),
        models={},
    )


@pytest.fixture(autouse=True)
def _reset():
    state._reset()
    yield
    state._reset()


# ---------- select_idle_candidates ----------
def test_select_idle_candidates_routing_idle_no_pending():
    state.set_status("m", ModelStatus.ROUTING, force=True)
    state._set_last_access("m", time.monotonic() - 120)
    assert background.select_idle_candidates(60, time.monotonic()) == ["m"]


def test_select_idle_candidates_excludes_non_routing():
    state.set_status("s", ModelStatus.STARTING, force=True)
    state._set_last_access("s", time.monotonic() - 120)
    assert background.select_idle_candidates(60, time.monotonic()) == []


def test_select_idle_candidates_excludes_pending():
    state.set_status("m", ModelStatus.ROUTING, force=True)
    state._set_last_access("m", time.monotonic() - 120)
    state.inc_pending("m")
    assert background.select_idle_candidates(60, time.monotonic()) == []


def test_select_idle_candidates_excludes_not_yet_idle():
    state.set_status("m", ModelStatus.ROUTING, force=True)
    state._set_last_access("m", time.monotonic())
    assert background.select_idle_candidates(60, time.monotonic()) == []


# ---------- idle_reclamation_loop ----------
async def test_idle_loop_reclaims_stale_routing():
    state.set_status("m", ModelStatus.ROUTING, force=True)
    state._set_last_access("m", time.monotonic() - 120)
    life = _FakeLife()
    ev = asyncio.Event()
    task = asyncio.create_task(
        background.idle_reclamation_loop(life, lambda: _alive_cfg(1), ev, period=0.01)
    )
    await asyncio.sleep(0.05)
    ev.set()
    await task
    assert life.stopped == ["m"]


async def test_idle_loop_skips_when_pending_at_scan():
    state.set_status("m", ModelStatus.ROUTING, force=True)
    state._set_last_access("m", time.monotonic() - 120)
    state.inc_pending("m")
    life = _FakeLife()
    ev = asyncio.Event()
    task = asyncio.create_task(
        background.idle_reclamation_loop(life, lambda: _alive_cfg(1), ev, period=0.01)
    )
    await asyncio.sleep(0.05)
    ev.set()
    await task
    assert life.stopped == []


async def test_idle_loop_double_check_skips_new_pending():
    state.set_status("a", ModelStatus.ROUTING, force=True)
    state.set_status("b", ModelStatus.ROUTING, force=True)
    state._set_last_access("a", time.monotonic() - 120)
    state._set_last_access("b", time.monotonic() - 120)

    def stop_during_a(name):
        if name == "a":
            state.inc_pending("b")

    life = _FakeLife(stop=stop_during_a)
    ev = asyncio.Event()
    task = asyncio.create_task(
        background.idle_reclamation_loop(life, lambda: _alive_cfg(1), ev, period=0.01)
    )
    await asyncio.sleep(0.05)
    ev.set()
    await task
    assert "a" in life.stopped
    assert "b" not in life.stopped


async def test_idle_loop_disabled_when_alive_sec_le_zero():
    state.set_status("m", ModelStatus.ROUTING, force=True)
    state._set_last_access("m", time.monotonic() - 120)
    life = _FakeLife()
    ev = asyncio.Event()
    task = asyncio.create_task(
        background.idle_reclamation_loop(life, lambda: _alive_cfg(0), ev, period=0.01)
    )
    await asyncio.sleep(0.05)
    ev.set()
    await task
    assert life.stopped == []


async def test_idle_loop_survives_stop_exception(caplog):
    state.set_status("m", ModelStatus.ROUTING, force=True)
    state._set_last_access("m", time.monotonic() - 120)

    def boom(name):
        raise RuntimeError("stop failed")

    life = _FakeLife(stop=boom)
    ev = asyncio.Event()
    task = asyncio.create_task(
        background.idle_reclamation_loop(life, lambda: _alive_cfg(1), ev, period=0.01)
    )
    await asyncio.sleep(0.05)
    ev.set()
    await task
    assert life.stopped == ["m"]


async def test_idle_loop_reads_fresh_alive_time_each_tick():
    state.set_status("m", ModelStatus.ROUTING, force=True)
    state._set_last_access("m", time.monotonic() - 120)
    cur = {"alive": 0}
    life = _FakeLife()
    ev = asyncio.Event()
    task = asyncio.create_task(
        background.idle_reclamation_loop(life, lambda: _alive_cfg(cur["alive"]), ev, period=0.01)
    )
    await asyncio.sleep(0.02)
    assert life.stopped == []
    cur["alive"] = 1
    await asyncio.sleep(0.05)
    ev.set()
    await task
    assert life.stopped == ["m"]


# ---------- auto_start ----------
def _auto_cfg(models_devs):
    from llm_node.config import ModelConfig, Scheme

    models = {
        name: ModelConfig(
            name,
            (name,),
            "Chat",
            i + 1,
            False,
            {"S": Scheme("S", frozenset({dev}), "a.bat", {dev: 1024})},
        )
        for i, (name, dev) in enumerate(models_devs)
    }
    return AppConfig(
        program=ProgramConfig(host="x", port=1, alive_time=60, log_level="INFO"),
        models=models,
    )


class _AutoDev:
    def __init__(self, online):
        self._online = set(online)
        self.refresh_calls = 0

    def online_devices(self):
        return set(self._online)

    def snapshot(self):
        return {}

    def refresh(self):
        self.refresh_calls += 1


async def test_auto_start_concurrent_all_models():
    life = _FakeLife()
    cfg = _auto_cfg([("a", "rtx 4060"), ("b", "rtx 4060"), ("c", "rtx 4060")])
    await background.auto_start(
        life, ["a", "b", "c"], cfg, _AutoDev({"rtx 4060"}), timeout=1.0, stop_event=asyncio.Event()
    )
    assert sorted(life.started) == ["a", "b", "c"]


async def test_auto_start_timeout_does_not_raise(caplog):
    async def slow(name):
        await asyncio.sleep(10)
        return ModelStatus.ROUTING

    life = _FakeLife(ensure_running=slow)
    cfg = _auto_cfg([("x", "rtx 4060")])
    with caplog.at_level(logging.WARNING):
        await background.auto_start(
            life, ["x"], cfg, _AutoDev({"rtx 4060"}), timeout=0.05, stop_event=asyncio.Event()
        )
    assert any("timeout" in r.message for r in caplog.records)


async def test_auto_start_failure_does_not_raise(caplog):
    def boom(name):
        raise RuntimeError("ensure boom")

    life = _FakeLife(ensure_running=boom)
    cfg = _auto_cfg([("x", "rtx 4060")])
    with caplog.at_level(logging.ERROR):
        await background.auto_start(
            life, ["x"], cfg, _AutoDev({"rtx 4060"}), timeout=1.0, stop_event=asyncio.Event()
        )
    assert any("failed" in r.message for r in caplog.records)


async def test_auto_start_empty_models_noop():
    life = _FakeLife()
    await background.auto_start(
        life, [], _auto_cfg([]), _AutoDev(set()), timeout=1.0, stop_event=asyncio.Event()
    )
    assert life.started == []


async def test_auto_start_device_isolation_batches():
    cfg = _auto_cfg([("m_rtx", "rtx 4060"), ("m_apu", "780m"), ("m_rtx2", "rtx 4060")])
    order: list = []

    class _OrderLife:
        async def ensure_running(self, name):
            order.append(name)
            return ModelStatus.ROUTING

    await background.auto_start(
        _OrderLife(),
        ["m_rtx", "m_apu", "m_rtx2"],
        cfg,
        _AutoDev({"rtx 4060", "780m"}),
        timeout=10.0,
        stop_event=asyncio.Event(),
    )
    assert set(order) == {"m_rtx", "m_apu", "m_rtx2"}
    assert order.index("m_rtx2") > order.index("m_rtx")
    assert order.index("m_rtx2") > order.index("m_apu")


# ---------- _plan_batches ----------
def test_plan_batches_device_isolation():
    from llm_node.config import Scheme
    from llm_node.runtime.background import _plan_batches

    s_rtx = Scheme("RTX", frozenset({"rtx 4060"}), "a.bat", {"rtx 4060": 5120})
    s_v100_780 = Scheme(
        "V100-780M", frozenset({"v100", "780m"}), "c.bat", {"v100": 0, "780m": 2048}
    )
    planned = [("qwen4b", s_rtx), ("qwen2b", s_v100_780), ("reranker", s_rtx)]
    parallel, serial = _plan_batches(planned)
    assert parallel == ["qwen4b", "qwen2b"]
    assert serial == ["reranker"]


def test_plan_batches_all_same_device_only_first_parallel():
    from llm_node.config import Scheme
    from llm_node.runtime.background import _plan_batches

    s = Scheme("S", frozenset({"rtx 4060"}), "a.bat", {"rtx 4060": 5120})
    assert _plan_batches([("a", s), ("b", s), ("c", s)]) == (["a"], ["b", "c"])


def test_plan_batches_empty():
    from llm_node.runtime.background import _plan_batches

    assert _plan_batches([]) == ([], [])

import asyncio
import time as _time
from pathlib import Path

import pytest

from llm_node import model_log, state
from llm_node.config import AppConfig, ModelConfig, ProgramConfig, Scheme
from llm_node.devices import DeviceInfo
from llm_node.probes import ProbeResult
from llm_node.runtime.lifecycle import Lifecycle
from llm_node.state import ModelStatus
from llm_node.supervisor import ProcessRecord


class FakeSupervisor:
    def __init__(self):
        self.spawned: list[object] = []
        self.killed: list[int] = []
        self.next_pid = 1000
        self.alive_pids: set[int] = set()
        self.exit_cbs: dict[int, object] = {}
        self.spawn_raises: Exception | None = None

    async def spawn(self, cmd, *, shell=True, on_output=None, env=None, cwd=None):
        if self.spawn_raises:
            exc, self.spawn_raises = self.spawn_raises, None
            raise exc
        pid = self.next_pid
        self.next_pid += 1
        self.spawned.append((cmd, shell))
        self.alive_pids.add(pid)
        return ProcessRecord(pid=pid, started_at=0.0)

    async def kill_tree(self, pid):
        self.killed.append(pid)
        self.alive_pids.discard(pid)
        return True

    def alive(self, pid):
        return pid in self.alive_pids

    def on_exit(self, pid, cb):
        self.exit_cbs[pid] = cb

    def trigger_exit(self, pid, code=-1):
        self.alive_pids.discard(pid)
        cb = self.exit_cbs.get(pid)
        if cb:
            cb(code)


def _dev(name, avail, total=8192):
    return DeviceInfo(name, "GPU", "VRAM", total, avail, total - avail, 0.0, None)


class FakeDevices:
    def __init__(self, online=None, snap=None):
        self._online = set(online) if online else {"rtx 4060"}
        self._snap = dict(snap) if snap is not None else {"rtx 4060": _dev("rtx 4060", 8192)}
        self.freed_mb: dict[str, int] = {}

    def online_devices(self):
        return set(self._online)

    def snapshot(self):
        out = {}
        for dev, info in self._snap.items():
            freed = self.freed_mb.get(dev, 0)
            if freed:
                out[dev] = DeviceInfo(
                    info.device_name,
                    info.device_type,
                    info.memory_type,
                    info.total_memory_mb,
                    info.available_memory_mb + freed,
                    max(0, info.used_memory_mb - freed),
                    info.usage_percentage,
                    info.temperature_celsius,
                )
            else:
                out[dev] = info
        return out

    def refresh(self):
        pass


def _model(name="m1", mode="Chat", port=8000, dev="rtx 4060", mem=2048):
    return ModelConfig(
        primary_name=name,
        aliases=(name,),
        mode=mode,
        port=port,
        schemes={
            "s": Scheme(
                config_source="s",
                required_devices=frozenset({dev}),
                script_path="run.cmd",
                memory_mb={dev: mem},
            )
        },
    )


def _cfg(*models):
    return AppConfig(
        program=ProgramConfig(host="127.0.0.1", port=8080, alive_time=60, log_level="INFO"),
        models={m.primary_name: m for m in models},
    )


def _ok_probe(alias, port, start_time=None, timeout=60):
    return ProbeResult(True, "ok")


def _make(sup=None, dev=None, probes=None, models=None):
    sup = sup or FakeSupervisor()
    dev = dev or FakeDevices()
    probes = probes if probes is not None else {"Chat": _ok_probe}
    cfg = _cfg(*(models if models is not None else [_model()]))
    return (
        Lifecycle(get_cfg=lambda: cfg, supervisor=sup, devices=dev, probes=probes),
        sup,
        dev,
        cfg,
    )


@pytest.fixture(autouse=True)
def _reset():
    state._reset()
    model_log.reset()
    yield
    state._reset()
    model_log.reset()


# ---------- cold start ----------
async def test_cold_start_reaches_routing():
    life, sup, _, _ = _make()
    status = await life.ensure_running("m1")
    assert status == ModelStatus.ROUTING
    assert len(sup.spawned) == 1
    assert state.get_pid("m1") == 1000
    assert "m1" in life._active_schemes
    assert sup.spawned[0][0] == "run.cmd"  # script_path
    assert sup.spawned[0][1] is True  # shell=True


# ---------- reconcile ----------
async def test_reconcile_dead_process_in_routing_marks_failed():
    life, sup, _, _ = _make()
    await life.ensure_running("m1")
    sup.alive_pids.discard(1000)
    status = await life.ensure_running("m1")
    assert status == ModelStatus.ROUTING  # reconcile 修正 FAILED 后重启到 ROUTING


async def test_reconcile_orphan_starting_with_no_inflight_marks_failed():
    life, _sup, _, _ = _make()
    state.set_status("m1", ModelStatus.STARTING, force=True)
    assert state.has_inflight("m1") is False
    status = await life.ensure_running("m1")
    assert status == ModelStatus.ROUTING


# ---------- on_crash ----------
async def test_external_crash_marks_failed_then_restart():
    life, sup, _, _ = _make()
    await life.ensure_running("m1")
    sup.trigger_exit(1000, code=1)
    assert state.get_status("m1") == ModelStatus.FAILED
    assert "exited code=1" in (state.get_failure_reason("m1") or "")
    status = await life.ensure_running("m1")
    assert status == ModelStatus.ROUTING


async def test_cooperative_stop_exit_is_not_marked_failed():
    life, sup, _, _ = _make()
    await life.ensure_running("m1")
    await life.stop("m1")
    sup.trigger_exit(1000, code=0)
    assert state.get_status("m1") == ModelStatus.STOPPED


# ---------- stop ----------
async def test_force_stop_then_restart_core_requirement():
    life, sup, _, _ = _make()
    await life.ensure_running("m1")
    assert state.get_status("m1") == ModelStatus.ROUTING
    await life.stop("m1")
    assert state.get_status("m1") == ModelStatus.STOPPED
    assert state.get_pid("m1") is None
    status = await life.ensure_running("m1")
    assert status == ModelStatus.ROUTING
    assert len(sup.spawned) == 2


async def test_stop_on_stopped_or_failed_is_noop():
    life, sup, _, _ = _make()
    s = await life.stop("m1")
    assert s == ModelStatus.STOPPED
    assert sup.killed == []
    state.record_failure("m1", "x")
    s2 = await life.stop("m1")
    assert s2 == ModelStatus.FAILED
    assert sup.killed == []


async def test_stop_from_each_running_state_lands_stopped():
    life, _sup, _, _ = _make()
    for pre in (
        ModelStatus.STARTING,
        ModelStatus.INIT_SCRIPT,
        ModelStatus.HEALTH_CHECK,
        ModelStatus.ROUTING,
        ModelStatus.FAILED,
    ):
        state._reset()
        life._stop_events.pop("m1", None)
        life._active_schemes.pop("m1", None)
        state.set_status("m1", pre, force=True)
        s = await life.stop("m1")
        assert s == ModelStatus.STOPPED or (pre == ModelStatus.FAILED and s == ModelStatus.FAILED)


# ---------- single-dispatch / checkpoints / errors / race / eviction ----------
async def test_single_dispatch_concurrent_start_spawns_once():
    life, sup, _, _ = _make()
    s1, s2 = await asyncio.gather(life.ensure_running("m1"), life.ensure_running("m1"))
    assert s1 == s2 == ModelStatus.ROUTING
    assert len(sup.spawned) == 1


async def test_stop_starting_winner_self_terminates_no_routing():
    def slow_probe(alias, port, start_time=None, timeout=60):
        _time.sleep(0.15)
        return ProbeResult(False, "slow")

    life, _sup, _, _ = _make(probes={"Chat": slow_probe})
    task = asyncio.create_task(life.ensure_running("m1"))
    await asyncio.sleep(0.05)
    await life.stop("m1")
    status = await task
    assert status == ModelStatus.STOPPED
    assert state.get_status("m1") == ModelStatus.STOPPED


async def test_slow_probe_then_concurrent_restart_not_clobbered():
    def slow_probe(alias, port, start_time=None, timeout=60):
        _time.sleep(0.3)
        return ProbeResult(True, "ok")

    life, sup, _, _ = _make(probes={"Chat": slow_probe})
    w1 = asyncio.create_task(life.ensure_running("m1"))
    await asyncio.sleep(0.05)
    await life.stop("m1")
    restart_status = await life.ensure_running("m1")
    await w1
    assert restart_status == ModelStatus.ROUTING
    assert state.get_status("m1") == ModelStatus.ROUTING
    assert len(sup.spawned) == 2


async def test_post_spawn_stop_kills_orphan_no_leak():
    life, sup, _, _ = _make()
    orig_spawn = sup.spawn

    async def spy_spawn(cmd, *, shell=False, on_output=None, env=None, cwd=None):
        rec = await orig_spawn(cmd, shell=shell, on_output=on_output)
        life._stop_events["m1"].set()
        return rec

    sup.spawn = spy_spawn
    status = await life.ensure_running("m1")
    assert status == ModelStatus.STOPPED
    assert 1000 in sup.killed


async def test_no_scheme_marks_failed():
    life, _sup, _, _ = _make(dev=FakeDevices(online=set(), snap={}))
    status = await life.ensure_running("m1")
    assert status == ModelStatus.FAILED


async def test_insufficient_resource_marks_failed():
    dev = FakeDevices(online={"rtx 4060"}, snap={"rtx 4060": _dev("rtx 4060", 0)})
    life, _sup, _dev2, _ = _make(dev=dev, models=[_model("m1", mem=4096)])
    status = await life.ensure_running("m1")
    assert status == ModelStatus.FAILED


async def test_probe_failure_marks_failed():
    def bad_probe(alias, port, start_time=None, timeout=60):
        return ProbeResult(False, "unhealthy")

    life, sup, _, _ = _make(probes={"Chat": bad_probe})
    status = await life.ensure_running("m1")
    assert status == ModelStatus.FAILED
    assert 1000 in sup.killed


async def test_probe_timeout_marks_failed():
    def timeout_probe(alias, port, start_time=None, timeout=60):
        _time.sleep(0.1)
        return ProbeResult(False, "探测器深层检查超时")

    life, _sup, _, _ = _make(probes={"Chat": timeout_probe})
    status = await life.ensure_running("m1")
    assert status == ModelStatus.FAILED


async def test_probe_raising_after_spawn_kills_pid_then_failed():
    def raising_probe(alias, port, start_time=None, timeout=60):
        raise RuntimeError("probe blew up")

    life, sup, _, _ = _make(probes={"Chat": raising_probe})
    status = await life.ensure_running("m1")
    assert status == ModelStatus.FAILED
    assert 1000 in sup.killed


async def test_spawn_exception_marks_failed_no_future_leak():
    life, sup, _, _ = _make()
    sup.spawn_raises = RuntimeError("boom")
    status = await life.ensure_running("m1")
    assert status == ModelStatus.FAILED
    assert state.has_inflight("m1") is False


async def test_pipeline_midstage_exception_clears_inflight():
    class BoomDevices(FakeDevices):
        def refresh(self):
            raise RuntimeError("nvidia-smi died")

    life, _sup, _, _ = _make(dev=BoomDevices())
    status = await life.ensure_running("m1")
    assert status == ModelStatus.FAILED
    assert state.has_inflight("m1") is False


async def test_eviction_executed_then_cold_start_reaches_routing():
    m1 = _model("m1", port=8000, mem=2048)
    m2 = _model("m2", port=8001, mem=4096)
    sup = FakeSupervisor()
    dev = FakeDevices(online={"rtx 4060"}, snap={"rtx 4060": _dev("rtx 4060", 2048, total=4096)})
    orig_kill = sup.kill_tree

    async def kill_releases(pid, *a, **kw):
        r = await orig_kill(pid, *a, **kw)
        dev.freed_mb["rtx 4060"] = dev.freed_mb.get("rtx 4060", 0) + 2048
        return r

    sup.kill_tree = kill_releases
    life, _, _, _ = _make(sup=sup, dev=dev, models=[m1, m2])
    await life.ensure_running("m1")
    m1_pid = state.get_pid("m1")
    status = await life.ensure_running("m2")
    assert status == ModelStatus.ROUTING
    assert m1_pid in sup.killed
    assert state.get_status("m2") == ModelStatus.ROUTING


def test_illegal_transition_raises_value_error():
    state._reset()
    state.set_status("m1", ModelStatus.STARTING, force=True)
    state.set_status("m1", ModelStatus.INIT_SCRIPT)
    with pytest.raises(ValueError):
        state.set_status("m1", ModelStatus.ROUTING)


# ---------- unload_all ----------
async def test_unload_all_stops_running_models():
    life, _sup, _, _ = _make(models=[_model("m1", port=8000), _model("m2", port=8001)])
    await life.ensure_running("m1")
    await life.ensure_running("m2")
    stopped = await life.unload_all()
    assert set(stopped) == {"m1", "m2"}
    assert state.get_status("m1") == ModelStatus.STOPPED
    assert state.get_status("m2") == ModelStatus.STOPPED


async def test_unload_all_skips_already_stopped():
    life, _sup, _, _ = _make(models=[_model("m1", port=8000)])
    stopped = await life.unload_all()
    assert stopped == []


async def test_unload_all_tolerates_one_stop_failure():
    life, sup, _, _ = _make(models=[_model("m1", port=8000), _model("m2", port=8001)])
    await life.ensure_running("m1")
    await life.ensure_running("m2")
    bad_pid = state.get_pid("m2")

    async def kill_tree(pid):
        if pid == bad_pid:
            raise RuntimeError("kill boom")
        sup.killed.append(pid)
        sup.alive_pids.discard(pid)
        return True

    sup.kill_tree = kill_tree
    stopped = await life.unload_all()
    assert "m1" in stopped
    assert "m2" not in stopped
    assert state.get_status("m1") == ModelStatus.STOPPED


# ---------- cancel-safe ----------
async def test_ensure_running_cancelled_after_spawn_kills_pid_clears_slot():
    def slow_probe(alias, port, start_time=None, timeout=60):
        _time.sleep(0.3)
        return ProbeResult(True, "ok")

    life, sup, _, _ = _make(probes={"Chat": slow_probe})
    task = asyncio.create_task(life.ensure_running("m1"))
    await asyncio.sleep(0.05)
    assert state.get_pid("m1") is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert 1000 in sup.killed
    assert state.has_inflight("m1") is False
    assert state.get_status("m1") == ModelStatus.FAILED


# ---------- spawn lock ----------
async def test_spawn_lock_serializes_concurrent_spawns():
    import time as _t

    sup = FakeSupervisor()
    spawn_log: list = []
    _real_spawn = sup.spawn

    async def logged_spawn(cmd, *, shell=False, on_output=None, env=None, cwd=None):
        spawn_log.append(("start", _t.monotonic()))
        await asyncio.sleep(0.05)
        rec = await _real_spawn(cmd, shell=shell, on_output=on_output)
        spawn_log.append(("end", _t.monotonic()))
        return rec

    sup.spawn = logged_spawn

    models = [_model("a", dev="rtx 4060"), _model("b", dev="780m")]
    life, _sup, _d, _c = _make(
        sup=sup,
        dev=FakeDevices(
            online={"rtx 4060", "780m"},
            snap={"rtx 4060": _dev("rtx 4060", 8192), "780m": _dev("780m", 8192)},
        ),
        models=models,
        probes={"Chat": _ok_probe},
    )
    await asyncio.gather(life.ensure_running("a"), life.ensure_running("b"))
    assert spawn_log[0][0] == "start"
    assert spawn_log[1][0] == "end"
    assert spawn_log[2][0] == "start"


async def test_spawn_lock_preserves_inflight_eviction_protection():
    models = [_model("a", dev="rtx 4060", mem=4096), _model("b", dev="rtx 4060", mem=8192)]
    life, sup, _d, _c = _make(
        sup=FakeSupervisor(),
        dev=FakeDevices(online={"rtx 4060"}, snap={"rtx 4060": _dev("rtx 4060", 4096)}),
        models=models,
        probes={"Chat": _ok_probe},
    )
    await life.ensure_running("a")
    state.begin_request("a")
    status = await life.ensure_running("b")
    assert "a" not in sup.killed
    assert status == ModelStatus.FAILED
    state.end_request("a")


async def test_ensure_running_inc_pending_closes_idle_reclaim_tocou():
    life, _sup, _d, _c = _make()
    await life.ensure_running("m1")
    assert state.get_status("m1") == ModelStatus.ROUTING
    assert state.pending_count("m1") == 0

    status = await life.ensure_running("m1", inc_pending=True)
    assert status == ModelStatus.ROUTING
    assert state.pending_count("m1") == 1

    state.end_request("m1")
    assert state.pending_count("m1") == 0


async def test_ensure_running_inc_pending_skips_when_not_routing():
    life, _sup, _d, _c = _make(probes={"Chat": lambda *a, **k: ProbeResult(False, "fail")})
    status = await life.ensure_running("m1", inc_pending=True)
    assert status == ModelStatus.FAILED
    assert state.pending_count("m1") == 0


# ---------- model_log wiring ----------
class _CapturingSupervisor(FakeSupervisor):
    on_output: object | None

    def __init__(self):
        super().__init__()
        self.on_output = None

    async def spawn(self, cmd, *, shell=True, on_output=None, env=None, cwd=None):
        self.on_output = on_output
        return await super().spawn(cmd, shell=shell, on_output=on_output)


async def test_pipeline_wires_on_output_to_model_log(tmp_path, monkeypatch):
    import llm_node.model_log as ml

    monkeypatch.setattr(ml, "_BASE_DIR", str(tmp_path))
    model_log.reset()
    cap = _CapturingSupervisor()
    lc, _sup, _, _ = _make(sup=cap)
    await lc.ensure_running("m1")
    assert cap.on_output is not None
    sid = model_log.resolve_session("m1")
    assert sid is not None
    cap.on_output("server listening on :8000", "out")
    cap.on_output("error: boom", "err")
    path = model_log.session_path("m1")
    text = await asyncio.to_thread(Path(path).read_text)
    assert "server listening on :8000" in text
    assert "error: boom" in text
    model_log.reset()


async def test_stop_ends_log_session(tmp_path, monkeypatch):
    import llm_node.model_log as ml

    monkeypatch.setattr(ml, "_BASE_DIR", str(tmp_path))
    model_log.reset()
    cap = _CapturingSupervisor()
    lc, _sup, _, _ = _make(sup=cap)
    await lc.ensure_running("m1")
    cap.on_output("old session line", "out")
    assert model_log.resolve_session("m1") is not None
    await lc.stop("m1")
    assert model_log.resolve_session("m1") is None
    model_log.capture("m1", "dropped after stop", "out")
    path = model_log.session_path("m1")
    assert path is None
    model_log.reset()


# ---------- get_cfg read-through ----------
async def test_lifecycle_reads_fresh_cfg_each_call():
    current = {"cfg": _cfg(_model("m1", port=8000))}
    life = Lifecycle(
        get_cfg=lambda: current["cfg"],
        supervisor=FakeSupervisor(),
        devices=FakeDevices(),
        probes={"Chat": _ok_probe},
    )
    assert "m1" in life._get_cfg().models
    current["cfg"] = _cfg(_model("m1", port=8000), _model("m2", port=8001))
    assert set(life._get_cfg().models) == {"m1", "m2"}
    state.set_status("m2", ModelStatus.ROUTING, force=True)
    state.record_pid("m2", 42)
    runnable = life._runnable(exclude="m1")
    assert "m2" in runnable


# ---------- script_path substitution ----------
async def test_pipeline_substitutes_vars_in_script_path():
    m = ModelConfig(
        primary_name="m1",
        aliases=("m1-served",),
        mode="Chat",
        port=12345,
        schemes={
            "s": Scheme(
                "s",
                frozenset({"rtx 4060"}),
                script_path="run_{{port}}_{{alias}}.sh",
                memory_mb={"rtx 4060": 2048},
            )
        },
    )
    life, sup, _, _ = _make(models=[m])
    await life.ensure_running("m1")
    assert sup.spawned[0][0] == "run_12345_m1-served.sh"


async def test_pipeline_normalizes_script_path_on_windows(monkeypatch):
    import os as _os

    m = ModelConfig(
        primary_name="m1",
        aliases=("m1",),
        mode="Chat",
        port=1000,
        schemes={
            "s": Scheme(
                "s",
                frozenset({"rtx 4060"}),
                script_path="Model_startup_script/run.bat",
                memory_mb={"rtx 4060": 2048},
            )
        },
    )
    monkeypatch.setattr(_os, "name", "nt")
    life, sup, _, _ = _make(models=[m])
    await life.ensure_running("m1")
    assert sup.spawned[0][0] == _os.path.normpath("Model_startup_script/run.bat")


async def test_pipeline_keeps_script_path_on_posix(monkeypatch):
    import os as _os

    m = ModelConfig(
        primary_name="m1",
        aliases=("m1",),
        mode="Chat",
        port=1000,
        schemes={
            "s": Scheme(
                "s",
                frozenset({"rtx 4060"}),
                script_path="Model_startup_script/run.sh",
                memory_mb={"rtx 4060": 2048},
            )
        },
    )
    monkeypatch.setattr(_os, "name", "posix")
    life, sup, _, _ = _make(models=[m])
    await life.ensure_running("m1")
    assert sup.spawned[0][0] == "Model_startup_script/run.sh"

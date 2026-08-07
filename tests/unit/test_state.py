import asyncio

import pytest

from llm_node.state import (
    ModelStatus,
    _reset,
    begin_request,
    claim_start,
    dec_pending,
    end_request,
    finish_start,
    get_status,
    inc_pending,
    is_failed,
    is_runnable,
    pending_count,
    record_failure,
    set_status,
)


@pytest.fixture(autouse=True)
def _clean():
    _reset()
    yield
    _reset()


def test_status_enum_six_states():
    assert {s.value for s in ModelStatus} == {
        "stopped",
        "starting",
        "init_script",
        "health_check",
        "routing",
        "failed",
    }


def test_default_status_is_stopped():
    assert get_status("M") == ModelStatus.STOPPED


def test_happy_path_transitions():
    set_status("M", ModelStatus.STARTING)
    set_status("M", ModelStatus.INIT_SCRIPT)
    set_status("M", ModelStatus.HEALTH_CHECK)
    set_status("M", ModelStatus.ROUTING)
    assert is_runnable("M")


def test_failure_transition_sets_reason():
    set_status("M", ModelStatus.STARTING)
    set_status("M", ModelStatus.FAILED, reason="boom")
    assert is_failed("M")


def test_failed_to_start_retry_is_legal():
    set_status("M", ModelStatus.STARTING)
    set_status("M", ModelStatus.FAILED, reason="x")
    set_status("M", ModelStatus.STARTING)
    assert get_status("M") == ModelStatus.STARTING


def test_illegal_transition_raises():
    set_status("M", ModelStatus.STARTING)
    set_status("M", ModelStatus.INIT_SCRIPT)
    with pytest.raises(ValueError):
        set_status("M", ModelStatus.ROUTING)  # INIT_SCRIPT→ROUTING illegal


def test_force_stop_from_any_state():
    for target in (
        ModelStatus.STARTING,
        ModelStatus.INIT_SCRIPT,
        ModelStatus.HEALTH_CHECK,
        ModelStatus.ROUTING,
        ModelStatus.FAILED,
    ):
        _reset()
        set_status("M", ModelStatus.STARTING)
        if target == ModelStatus.INIT_SCRIPT:
            set_status("M", ModelStatus.INIT_SCRIPT)
        elif target == ModelStatus.HEALTH_CHECK:
            set_status("M", ModelStatus.INIT_SCRIPT)
            set_status("M", ModelStatus.HEALTH_CHECK)
        elif target == ModelStatus.ROUTING:
            set_status("M", ModelStatus.INIT_SCRIPT)
            set_status("M", ModelStatus.HEALTH_CHECK)
            set_status("M", ModelStatus.ROUTING)
        elif target == ModelStatus.FAILED:
            set_status("M", ModelStatus.FAILED, reason="x")
        set_status("M", ModelStatus.STOPPED, force=True)
        assert get_status("M") == ModelStatus.STOPPED


def test_record_failure_helper():
    set_status("M", ModelStatus.STARTING)
    record_failure("M", "segfault")
    assert is_failed("M")


def test_pending_inc_dec_clamped():
    inc_pending("M")
    inc_pending("M")
    assert pending_count("M") == 2
    dec_pending("M")
    dec_pending("M")
    dec_pending("M")
    assert pending_count("M") == 0


def test_begin_end_request_touch_activity():
    from llm_node.state import _state

    begin_request("M")
    assert pending_count("M") == 1
    t1 = _state["M"].last_access
    end_request("M")
    assert pending_count("M") == 0
    assert _state["M"].last_access >= t1


def test_claim_start_winner_runs_loser_awaits():
    async def main():
        fut_a, won_a = claim_start("M")
        fut_b, won_b = claim_start("M")
        assert won_a is True and won_b is False
        assert fut_a is fut_b
        assert not fut_a.done()
        finish_start("M", ModelStatus.ROUTING)
        assert fut_a.result() == ModelStatus.ROUTING
        assert fut_b.result() == ModelStatus.ROUTING
        from llm_node.state import _inflight

        assert "M" not in _inflight

    asyncio.run(main())


def test_claim_start_after_finish_allows_new_start():
    async def main():
        fut, won = claim_start("M")
        assert won
        finish_start("M", ModelStatus.FAILED)
        fut2, won2 = claim_start("M")
        assert won2 is True
        assert fut2 is not fut

    asyncio.run(main())


def test_pid_accessors():
    from llm_node import state

    state._reset()
    state.record_pid("m1", 1234)
    assert state.get_pid("m1") == 1234
    state.clear_pid("m1")
    assert state.get_pid("m1") is None


def test_inflight_introspection_and_release():
    asyncio.run(_inflight_body())


async def _inflight_body():
    from llm_node import state

    state._reset()
    assert state.has_inflight("m1") is False
    fut, won = state.claim_start("m1")
    assert won is True
    assert state.has_inflight("m1") is True
    popped = state.pop_inflight("m1")
    assert popped is fut
    assert state.has_inflight("m1") is False
    assert state.pop_inflight("m1") is None
    state.claim_start("m2")
    state.clear_inflight("m2")
    assert state.has_inflight("m2") is False


def test_finish_start_owner_guard_no_clobber():
    asyncio.run(_owner_guard_body())


async def _owner_guard_body():
    from llm_node import state

    state._reset()
    fut1, _won = state.claim_start("m1")
    state.pop_inflight("m1")
    fut2, _ = state.claim_start("m1")
    state.finish_start("m1", ModelStatus.STOPPED, owner=fut1)
    assert state.get_status("m1") == ModelStatus.STARTING
    state.finish_start("m1", ModelStatus.ROUTING, owner=fut2)
    assert state.get_status("m1") == ModelStatus.ROUTING


def test_get_last_access():
    from llm_node import state

    state._reset()
    assert state.get_last_access("m1") == 0.0
    state.touch_activity("m1")
    assert state.get_last_access("m1") > 0.0


def test_get_failure_reason():
    from llm_node import state

    state._reset()
    assert state.get_failure_reason("m1") is None
    state.record_failure("m1", "exited code=1")
    assert state.get_failure_reason("m1") == "exited code=1"


def test_failure_reason_cleared_on_restart():
    from llm_node import state

    state._reset()
    state.record_failure("m1", "probe failed")
    assert state.get_failure_reason("m1") == "probe failed"
    asyncio.run(_claim_and_check_cleared())


async def _claim_and_check_cleared():
    from llm_node import state

    _fut, won = state.claim_start("m1")
    assert won
    assert state.get_failure_reason("m1") is None
    state.finish_start("m1", ModelStatus.ROUTING)
    assert state.get_failure_reason("m1") is None
    state.record_failure("m1", "second crash")
    assert state.get_failure_reason("m1") == "second crash"
    _fut2, _ = state.claim_start("m1")
    assert state.get_failure_reason("m1") is None


def test_routing_names_returns_only_routing():
    from llm_node import state

    state._reset()
    state.set_status("a", ModelStatus.ROUTING, force=True)
    state.set_status("b", ModelStatus.STARTING, force=True)
    state.set_status("c", ModelStatus.STOPPED, force=True)
    assert state.routing_names() == ["a"]


def test_set_last_access_test_helper():
    from llm_node import state

    state._reset()
    state._set_last_access("m1", 1234.5)
    assert state.get_last_access("m1") == 1234.5


def test_record_failure_clears_stale_pid():
    from llm_node import state

    state._reset()
    state.set_status("m1", ModelStatus.ROUTING, force=True)
    state.record_pid("m1", 1234)
    assert state.get_pid("m1") == 1234
    state.record_failure("m1", "process exited code=1")
    assert state.get_status("m1") == ModelStatus.FAILED
    assert state.get_pid("m1") is None


def test_routing_records_started_at_and_last_access_wall():
    from llm_node import state

    state._reset()
    state.set_status("m1", ModelStatus.ROUTING, force=True)
    assert state.get_started_at("m1") is not None
    assert state.get_started_at("m1") > 0
    assert state.get_last_access_wall("m1") > 0


def test_started_at_none_unless_routing():
    from llm_node import state

    state._reset()
    assert state.get_started_at("m1") is None
    state.set_status("m1", ModelStatus.STARTING, force=True)
    assert state.get_started_at("m1") is None
    state.set_status("m1", ModelStatus.ROUTING, force=True)
    assert state.get_started_at("m1") is not None
    state.set_status("m1", ModelStatus.STOPPED, force=True)
    assert state.get_started_at("m1") is None


def test_record_failure_clears_started_at():
    from llm_node import state

    state._reset()
    state.set_status("m1", ModelStatus.ROUTING, force=True)
    state.record_failure("m1", "x")
    assert state.get_started_at("m1") is None


def test_touch_activity_updates_last_access_wall():
    from llm_node import state

    state._reset()
    state.set_status("m1", ModelStatus.ROUTING, force=True)
    wall1 = state.get_last_access_wall("m1")
    state.touch_activity("m1")
    assert state.get_last_access_wall("m1") >= wall1

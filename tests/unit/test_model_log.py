from llm_node import model_log


def test_start_resolve_capture_end(tmp_path, monkeypatch):
    import llm_node.model_log as ml

    monkeypatch.setattr(ml, "_BASE_DIR", str(tmp_path))
    model_log.reset()
    sid = model_log.start_session("model", model_name="m1", alias="m1-served")
    assert model_log.resolve_session("m1-served") == sid
    model_log.capture("m1-served", "hello", "out")
    model_log.capture("m1-served", "boom", "err")
    path = model_log.session_path("m1-served")
    assert path is not None
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "hello" in text and "boom" in text
    assert "[out]" in text and "[err]" in text
    model_log.end_session(sid)
    assert model_log.resolve_session("m1-served") is None
    model_log.reset()


def test_end_session_idempotent(tmp_path, monkeypatch):
    import llm_node.model_log as ml

    monkeypatch.setattr(ml, "_BASE_DIR", str(tmp_path))
    model_log.reset()
    model_log.end_session(999)  # 未知 id → no-op
    model_log.reset()


def test_capture_no_active_session_noop(tmp_path, monkeypatch):
    import llm_node.model_log as ml

    monkeypatch.setattr(ml, "_BASE_DIR", str(tmp_path))
    model_log.reset()
    model_log.capture("nope", "line", "out")  # 无会话 → 丢弃,不抛
    model_log.reset()


def test_start_session_uses_safe_name(tmp_path, monkeypatch):
    import llm_node.model_log as ml

    monkeypatch.setattr(ml, "_BASE_DIR", str(tmp_path))
    model_log.reset()
    sid = model_log.start_session("model", model_name="a/b:c", alias="x")
    path = model_log.session_path("x")
    assert "a_b_c" in path
    model_log.end_session(sid)
    model_log.reset()


def test_rotation_keeps_newest_ten(tmp_path, monkeypatch):
    import llm_node.model_log as ml

    monkeypatch.setattr(ml, "_BASE_DIR", str(tmp_path))
    model_log.reset()
    sids = [model_log.start_session("model", model_name="m1", alias=f"a{i}") for i in range(12)]
    import glob
    import os

    files = glob.glob(os.path.join(str(tmp_path), "m1", "*.log"))
    assert len(files) <= 10
    for sid in sids:
        model_log.end_session(sid)
    model_log.reset()

from pathlib import Path

from llm_node.config import (
    AppConfig,
    ModelConfig,
    ModelMode,
    ProgramConfig,
    Scheme,
    load,
    resolve_alias,
    select_adaptive,
    substitute_vars,
    validate,
)


def _write_cfg(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_parses_models_and_preserves_device_names(tmp_path):
    cfg_path = _write_cfg(
        tmp_path,
        """
program: {host: 0.0.0.0, port: 8080, alive_time: 60, log_level: INFO}
Local-Models:
  Qwen3-4B:
    aliases: ["Qwen3-4B"]
    mode: Chat
    port: 10001
    RTX4060:
      required_devices: ["RTX 4060"]
      script_path: "scripts/q.bat"
      memory_mb: {"RTX 4060": 5120}
""",
    )
    cfg = load(cfg_path)
    m = cfg.models["Qwen3-4B"]
    assert m.port == 10001
    assert m.mode == "Chat"
    assert "Qwen3-4B" in m.aliases
    scheme = m.schemes["RTX4060"]
    assert isinstance(scheme, Scheme)
    assert scheme.required_devices == frozenset({"RTX 4060"})  # 存储原样,匹配时归一化
    assert scheme.memory_mb == {"RTX 4060": 5120}
    assert scheme.script_path == "scripts/q.bat"


def test_load_flat_model_keys_without_wrapper(tmp_path):
    """旧节点格式:模型直接置顶(无 Local-Models 包装)也应能加载。"""
    cfg_path = _write_cfg(
        tmp_path,
        """
program: {host: 0.0.0.0, port: 8080, alive_time: 60, log_level: INFO}
jina-code-embeddings:
  aliases: ["jina-code-embeddings"]
  mode: Embedding
  port: 10001
  CPU:
    required_devices: ["CPU"]
    script_path: "run.sh"
    memory_mb: {CPU: 2048}
""",
    )
    cfg = load(cfg_path)
    assert "jina-code-embeddings" in cfg.models
    assert cfg.models["jina-code-embeddings"].mode == "Embedding"


def test_model_mode_values():
    assert {m.value for m in ModelMode} == {"Chat", "Embedding", "Reranker"}


def test_validate_flags_port_and_alias_clash_and_bad_mode():
    cfg = AppConfig(
        program=ProgramConfig(host="0.0.0.0", port=8080, alive_time=60, log_level="INFO"),
        models={
            "A": ModelConfig("A", ("x",), "Chat", 1, False, {}),
            "B": ModelConfig("B", ("x",), "Embedding", 1, False, {}),
            "C": ModelConfig("C", ("y",), "Bogus", 2, False, {}),
            "D": ModelConfig("D", ("z",), "Chat", 3, False, {}),
        },
    )
    errs = validate(cfg)
    assert any("Port 1 shared" in e for e in errs)
    assert any("Alias 'x' shared" in e for e in errs)
    assert any("mode 'Bogus'" in e for e in errs)
    assert any("no device scheme" in e for e in errs)


def test_validate_passes_clean_config():
    cfg = AppConfig(
        program=ProgramConfig(host="0.0.0.0", port=8080, alive_time=60, log_level="INFO"),
        models={
            "A": ModelConfig(
                "A",
                ("a",),
                "Chat",
                1,
                False,
                {"S": Scheme("S", frozenset({"gpu"}), "a.bat", {"gpu": 1})},
            )
        },
    )
    assert validate(cfg) == []


def test_validate_rejects_empty_script_path():
    cfg = AppConfig(
        program=ProgramConfig("0.0.0.0", 8080, 60, "INFO"),
        models={
            "M": ModelConfig(
                "M",
                ("m",),
                "Chat",
                1,
                False,
                {"S": Scheme("S", frozenset({"gpu"}), "", {"gpu": 1})},
            )
        },
    )
    errs = validate(cfg)
    assert any("empty script_path" in e for e in errs)


def test_select_adaptive_first_subset_wins():
    s_gpu = Scheme("GPU", frozenset({"gpu"}), "g.bat", {"gpu": 1})
    s_apu = Scheme("APU", frozenset({"apu"}), "a.bat", {"apu": 1})
    m = ModelConfig("M", ("M",), "Chat", 1, False, {"GPU": s_gpu, "APU": s_apu})
    assert select_adaptive(m, {"gpu"}).config_source == "GPU"
    assert select_adaptive(m, {"apu"}).config_source == "APU"
    assert select_adaptive(m, set()) is None


def test_resolve_alias_to_primary():
    cfg = AppConfig(
        program=ProgramConfig(host="0.0.0.0", port=8080, alive_time=60, log_level="INFO"),
        models={"Qwen3-4B": ModelConfig("Qwen3-4B", ("Qwen3-4B", "q4"), "Chat", 1)},
    )
    assert resolve_alias(cfg, "q4") == "Qwen3-4B"
    assert resolve_alias(cfg, "Qwen3-4B") == "Qwen3-4B"
    try:
        resolve_alias(cfg, "nope")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_referenced_devices_unions_required_and_memory_keys():
    from llm_node.config import referenced_devices

    s1 = Scheme("S1", frozenset({"rtx 4060", "v100"}), "a.bat", {"rtx 4060": 5120})
    s2 = Scheme("S2", frozenset({"780m"}), "b.bat", {"780m": 2048, "v100": 0})
    m = ModelConfig(
        primary_name="M", aliases=("m",), mode="Chat", port=1000, schemes={"S1": s1, "S2": s2}
    )
    cfg = AppConfig(
        program=ProgramConfig("0.0.0.0", 8080, 60, "INFO"),
        models={"M": m},
    )
    assert referenced_devices(cfg) == {"rtx 4060", "v100", "780m"}


def test_referenced_devices_empty_when_no_schemes():
    from llm_node.config import referenced_devices

    m = ModelConfig(primary_name="M", aliases=("m",), mode="Chat", port=1000)
    cfg = AppConfig(
        program=ProgramConfig("0.0.0.0", 8080, 60, "INFO"),
        models={"M": m},
    )
    assert referenced_devices(cfg) == set()


def test_validate_rejects_out_of_range_ports():
    def _prog(port):
        return ProgramConfig(host="0.0.0.0", port=port, alive_time=60, log_level="INFO")

    cfg = AppConfig(
        program=_prog(99999),
        models={
            "M": ModelConfig(
                "M",
                ("m",),
                "Chat",
                0,
                False,
                {"S": Scheme("S", frozenset({"gpu"}), "a.bat", {"gpu": 1})},
            )
        },
    )
    errs = validate(cfg)
    assert any("Program port 99999 out of range" in e for e in errs)
    assert any("port 0 out of range" in e for e in errs)


def test_validate_rejects_empty_alias_and_intra_model_duplicate():
    cfg = AppConfig(
        program=ProgramConfig("0.0.0.0", 8080, 60, "INFO"),
        models={
            "M": ModelConfig(
                "M",
                ("dup", "dup", ""),
                "Chat",
                1,
                False,
                {"S": Scheme("S", frozenset({"gpu"}), "a.bat", {"gpu": 1})},
            )
        },
    )
    errs = validate(cfg)
    assert any("duplicate alias 'dup'" in e for e in errs)
    assert any("empty alias" in e for e in errs)


# ---------- substitute_vars:脚本路径变量替换 ----------


def _model(port: int = 10004, aliases: tuple[str, ...] = ("Qwen3.5-2B", "q")) -> ModelConfig:
    return ModelConfig(
        "M",
        aliases,
        "Chat",
        port,
        False,
        {"S": Scheme("S", frozenset({"gpu"}), "a.bat", {"gpu": 1})},
    )


def test_substitute_vars_replaces_port_and_first_alias():
    m = _model(port=10004, aliases=("Qwen3.5-2B", "q"))
    assert substitute_vars("scripts/run_{{port}}.bat", m) == "scripts/run_10004.bat"
    assert substitute_vars("scripts/{{alias}}.sh", m) == "scripts/Qwen3.5-2B.sh"
    assert substitute_vars("{{alias}} vs q", m) == "Qwen3.5-2B vs q"


def test_substitute_vars_without_placeholder_unchanged():
    m = _model()
    assert substitute_vars("scripts/run.bat", m) == "scripts/run.bat"


def test_substitute_vars_empty_aliases_yields_empty_alias():
    m = _model(aliases=())
    assert substitute_vars("{{alias}}", m) == ""

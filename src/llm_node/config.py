"""Config: YAML load → frozen dataclasses. Device names stored as-is (normalized on match).

Stateless node: no DB, no billing, no WOL. Schemes use ``script_path`` (original
node approach) instead of the manager's structured ``command``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml

PROGRAM_DEFAULTS: dict[str, str] = {
    "host": "0.0.0.0",
    "port": "8080",
    "alive_time": "60",
    "log_level": "INFO",
}


class ModelMode(str, Enum):
    """Probe selector; string values are config/registry keys."""

    CHAT = "Chat"
    EMBEDDING = "Embedding"
    RERANKER = "Reranker"


@dataclass(frozen=True, slots=True)
class Scheme:
    config_source: str
    required_devices: frozenset[str]
    script_path: str
    memory_mb: dict[str, int]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    primary_name: str
    aliases: tuple[
        str, ...
    ]  # 有序:aliases[0]=主别名=下游 served name(lmdeploy --model-name / llama.cpp -a)
    mode: str
    port: int
    auto_start: bool = False
    schemes: dict[str, Scheme] = field(default_factory=dict)


# 启动脚本变量占位符:{{port}} / {{alias}}(alias=aliases[0])。双大括号——
# 单大括号与 JSON 参数冲突,双大括号在脚本路径中几乎不出现,安全。
SUBST_PLACEHOLDERS = ("{{port}}", "{{alias}}")


def substitute_vars(text: str, model: ModelConfig) -> str:
    """脚本路径变量替换:{{port}} → 模型端口,{{alias}} → 第一别名(下游 served name)。
    有占位符才替换,无则原样。"""
    alias = model.aliases[0] if model.aliases else ""
    return text.replace("{{port}}", str(model.port)).replace("{{alias}}", alias)


@dataclass(frozen=True, slots=True)
class ProgramConfig:
    host: str
    port: int
    alive_time: int
    log_level: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    program: ProgramConfig
    models: dict[str, ModelConfig]


def load(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    p = raw.get("program", {})
    program = ProgramConfig(
        host=p.get("host", PROGRAM_DEFAULTS["host"]),
        port=int(p.get("port", int(PROGRAM_DEFAULTS["port"]))),
        alive_time=int(p.get("alive_time", int(PROGRAM_DEFAULTS["alive_time"]))),
        log_level=p.get("log_level", PROGRAM_DEFAULTS["log_level"]),
    )
    # 兼容两种写法:Local-Models 包装键(Manager 格式)或模型直接置顶(旧节点格式)
    if "Local-Models" in raw:
        models_raw: dict = raw["Local-Models"]
    else:
        models_raw = {k: v for k, v in raw.items() if k != "program"}

    reserved = {"aliases", "mode", "port", "auto_start"}
    models: dict[str, ModelConfig] = {}
    for name, m in models_raw.items():
        schemes: dict[str, Scheme] = {}
        for key, val in m.items():
            if key in reserved or not isinstance(val, dict):
                continue
            schemes[key] = Scheme(
                config_source=key,
                required_devices=frozenset(val.get("required_devices", [])),
                script_path=val["script_path"],
                memory_mb={k: int(v) for k, v in val.get("memory_mb", {}).items()},
            )
        models[name] = ModelConfig(
            primary_name=name,
            aliases=tuple(m.get("aliases", [])),
            mode=m.get("mode", "Chat"),
            port=int(m["port"]),
            auto_start=bool(m.get("auto_start", False)),
            schemes=schemes,
        )
    return AppConfig(program=program, models=models)


def validate(cfg: AppConfig) -> list[str]:
    errors: list[str] = []
    if not 1 <= cfg.program.port <= 65535:
        errors.append(f"Program port {cfg.program.port} out of range (1-65535)")
    seen_ports: dict[int, str] = {}
    seen_aliases: dict[str, str] = {}
    valid_modes = {m.value for m in ModelMode}
    for name, m in cfg.models.items():
        if not name or not name.strip():
            errors.append("Model name is empty/blank")
        if not 1 <= m.port <= 65535:
            errors.append(f"Model '{name}' port {m.port} out of range (1-65535)")
        if m.port in seen_ports:
            errors.append(f"Port {m.port} shared by models '{seen_ports[m.port]}' and '{name}'")
        else:
            seen_ports[m.port] = name
        if not m.aliases:
            errors.append(f"Model '{name}' has no aliases")
        for a in m.aliases:
            if not a or not a.strip():
                errors.append(f"Model '{name}' has empty alias")
                continue
            if a in seen_aliases:
                if seen_aliases[a] == name:
                    errors.append(f"Model '{name}' has duplicate alias '{a}'")
                else:
                    errors.append(f"Alias '{a}' shared by models '{seen_aliases[a]}' and '{name}'")
            else:
                seen_aliases[a] = name
        if m.mode not in valid_modes:
            errors.append(
                f"Model '{name}' mode '{m.mode}' not supported (supported: {sorted(valid_modes)})"
            )
        if not m.schemes:
            errors.append(f"Model '{name}' has no device scheme")
        for sname, scheme in m.schemes.items():
            if not scheme.script_path:
                errors.append(f"Model '{name}' scheme '{sname}' has empty script_path")
    return errors


def select_adaptive(model: ModelConfig, online: set[str]) -> Scheme | None:
    for scheme in model.schemes.values():
        if scheme.required_devices <= online:
            return scheme
    return None


def referenced_devices(cfg: AppConfig) -> set[str]:
    """收集 config 引用过的全部设备名 = ∪ scheme.required_devices ∪ ∪ scheme.memory_mb.keys()。"""
    names: set[str] = set()
    for m in cfg.models.values():
        for scheme in m.schemes.values():
            names |= set(scheme.required_devices)
            names |= set(scheme.memory_mb)
    return names


def resolve_alias(cfg: AppConfig, alias: str) -> str:
    for name, m in cfg.models.items():
        if alias == name or alias in m.aliases:
            return name
    raise KeyError(alias)


def auto_start_models(cfg: AppConfig) -> list[str]:
    """配置中 auto_start=True 的模型名列表。"""
    return [n for n, m in cfg.models.items() if m.auto_start]

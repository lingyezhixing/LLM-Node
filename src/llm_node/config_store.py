"""YAML-backed config holder (stateless replacement for the manager's DB config_store).

``snapshot()`` returns the frozen AppConfig; ``reload()`` re-reads + validates the
YAML file (gateway/proxy read-through so edits take effect without restart).
"""

from __future__ import annotations

from pathlib import Path

from llm_node import config
from llm_node.config import AppConfig


class ConfigStore:
    """File-backed config holder. frozen snapshot() is the consumer interface."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._snapshot = self._load()

    def _load(self) -> AppConfig:
        cfg = config.load(self._path)
        errors = config.validate(cfg)
        if errors:
            raise ValueError("Invalid config:\n" + "\n".join(f"  - {e}" for e in errors))
        return cfg

    def snapshot(self) -> AppConfig:
        return self._snapshot

    def reload(self) -> AppConfig:
        self._snapshot = self._load()
        return self._snapshot

"""Root-logger setup(控制台 + 时间戳文件 + 清理)。app.py 组合根做接线。"""

from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path


def _cleanup_old_logs(log_dir: str, keep: int = 10) -> None:
    """保留最近 keep 个 llm-node_*.log(按 mtime),删旧的。"""
    files = sorted(
        Path(log_dir).glob("llm-node_*.log"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for stale in files[keep:]:
        try:
            stale.unlink()
        except OSError:
            pass


def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    """配置 root logger(可重配):控制台 + 每次启动一个时间戳文件(留 10 个)。"""
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric)
    console.setFormatter(fmt)
    root.addHandler(console)
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = (
            Path(log_dir) / f"llm-node_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"  # noqa: DTZ005 — 文件名时间戳,本地时间即可
        )
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(numeric)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        _cleanup_old_logs(log_dir, keep=10)
        logging.getLogger(__name__).info("logging to %s", log_file)
    except OSError:
        pass
    logging.getLogger("httpx").setLevel(logging.WARNING)

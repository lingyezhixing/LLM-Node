"""File-based model log sessions (stateless replacement for the manager's DB logs).

Writes each model's spawn output (stdout/stderr) to ``logs/model_logs/<model>/<ts>.log``,
rotating to keep the newest 10 files per model. Thread-safe via a single lock;
``capture`` is called from supervisor reader threads (via call_soon_threadsafe).

API mirrors the subset the lifecycle needs: ``start_session / capture / end_session /
resolve_session`` + ``reset`` (test seam). No DB, no SSE broadcast.
"""

from __future__ import annotations

import glob
import logging
import os
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

_BASE_DIR = "logs/model_logs"
_KEEP = 10

_lock = threading.Lock()
# alias -> {"sid": int, "path": str, "fh": file}
_sessions: dict[str, dict] = {}
_sid_counter = 0


def _safe_name(name: str) -> str:
    return name.replace(":", "_").replace("\\", "_").replace("/", "_").replace(os.sep, "_")


def reset() -> None:
    """Test seam: close + clear all sessions (like the manager's logs.reset)."""
    global _sid_counter
    with _lock:
        for s in _sessions.values():
            try:
                s["fh"].close()
            except Exception:  # noqa: BLE001, S110
                pass
        _sessions.clear()
        _sid_counter = 0


def start_session(type_: str, *, model_name: str, alias: str) -> int:
    """Open a new log session for a model. Returns session id (int)."""
    global _sid_counter
    with _lock:
        _sid_counter += 1
        sid = _sid_counter
        safe = _safe_name(model_name)
        model_dir = os.path.join(_BASE_DIR, safe)
        try:
            os.makedirs(model_dir, exist_ok=True)
        except OSError as e:
            logger.warning("create model log dir failed for %s: %s", model_name, e)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005 — 文件名时间戳,本地时间即可
        path = os.path.join(model_dir, f"{safe}_{ts}_{sid}.log")
        try:
            fh = open(path, "w", encoding="utf-8")  # noqa: SIM115 — 句柄跨会话保持打开
            fh.write(f"=== Log Start: {model_name} at {ts} ===\n")
            fh.flush()
            _cleanup_old_locked(model_dir)  # 新文件已计入 glob,精确保留 _KEEP 个
        except OSError as e:
            logger.warning("open model log file failed for %s: %s", model_name, e)
            return sid
        _sessions[alias] = {"sid": sid, "path": path, "fh": fh}
        return sid


def _cleanup_old_locked(model_dir: str) -> None:
    """Keep the newest _KEEP log files in a model dir (caller holds _lock).

    Windows 无法删除仍被打开的句柄 → 被淘汰文件的活跃会话先收口(close + 摘除),
    再删除文件,保证轮换语义一致(active session 的文件被轮换掉即结束该会话)。"""
    try:
        files = sorted(glob.glob(os.path.join(model_dir, "*.log")), key=os.path.getmtime)
        while len(files) > _KEEP:
            oldest = files.pop(0)
            for alias, s in list(_sessions.items()):
                if s["path"] == oldest:
                    try:
                        s["fh"].close()
                    except Exception:  # noqa: BLE001, S110
                        pass
                    del _sessions[alias]
                    break
            try:
                os.remove(oldest)
            except OSError:
                pass
    except OSError:
        pass


def capture(alias: str, line: str, stream: str) -> None:
    """Append a stdout/stderr line to the alias's active session file (if any)."""
    with _lock:
        s = _sessions.get(alias)
        if s is None:
            return
        ts = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005 — 日志时间戳,本地时间即可
        prefix = "out" if stream == "out" else "err"
        try:
            s["fh"].write(f"[{ts}] [{prefix}] {line}\n")
            s["fh"].flush()
        except OSError as e:
            logger.warning("write model log failed for %s: %s", alias, e)


def end_session(sid: int) -> None:
    """Close a session by id (idempotent no-op for unknown id)."""
    with _lock:
        for alias, s in list(_sessions.items()):
            if s["sid"] == sid:
                try:
                    s["fh"].close()
                except Exception:  # noqa: BLE001, S110
                    pass
                del _sessions[alias]
                return


def resolve_session(alias: str) -> int | None:
    """Active session id for an alias, or None."""
    with _lock:
        s = _sessions.get(alias)
        return s["sid"] if s else None


def session_path(alias: str) -> str | None:
    """Active session file path for an alias, or None."""
    with _lock:
        s = _sessions.get(alias)
        return s["path"] if s else None

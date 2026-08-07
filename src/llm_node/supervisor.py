"""Cross-platform process supervisor. Process-group/session isolation is an
INTERNAL invariant (Win CREATE_NEW_PROCESS_GROUP, POSIX start_new_session).
One asyncio wait-task per process replaces the legacy 5s poller. Blocking ops
(Popen, psutil.wait, killpg) run via asyncio.to_thread."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import psutil


@dataclass(frozen=True, slots=True)
class ProcessRecord:
    pid: int
    started_at: float
    exit_code: int | None = None


@runtime_checkable
class ProcessRunner(Protocol):
    async def spawn(
        self,
        cmd,
        *,
        shell: bool = False,
        on_output: Callable[[str, str], None] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ProcessRecord: ...
    async def kill_tree(self, pid: int) -> bool: ...
    def alive(self, pid: int) -> bool: ...
    def on_exit(self, pid: int, cb: Callable[[int], None]) -> None: ...


def _popen_kwargs() -> dict:
    kw: dict = {"text": True, "encoding": "utf-8", "errors": "replace"}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
        kw["close_fds"] = True
    return kw


class Supervisor:
    def __init__(self) -> None:
        self._procs: dict[int, subprocess.Popen] = {}
        self._wait_tasks: dict[int, asyncio.Task] = {}
        self._exit_cbs: dict[int, Callable[[int], None]] = {}
        self._readers: dict[int, list[threading.Thread]] = {}

    async def spawn(
        self,
        cmd,
        *,
        shell: bool = False,
        on_output: Callable[[str, str], None] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> ProcessRecord:
        loop = asyncio.get_running_loop()
        popen = await asyncio.to_thread(
            subprocess.Popen,
            cmd,
            shell=shell,
            stdout=(subprocess.PIPE if on_output is not None else None),
            stderr=(subprocess.PIPE if on_output is not None else None),
            env=env,
            cwd=cwd,
            **_popen_kwargs(),
        )
        self._procs[popen.pid] = popen
        self._wait_tasks[popen.pid] = asyncio.create_task(self._wait(popen.pid))
        if on_output is not None:
            threads = [
                threading.Thread(
                    target=self._pump, args=(popen.stdout, "out", loop, on_output), daemon=True
                ),
                threading.Thread(
                    target=self._pump, args=(popen.stderr, "err", loop, on_output), daemon=True
                ),
            ]
            self._readers[popen.pid] = threads
            for t in threads:
                t.start()
        return ProcessRecord(pid=popen.pid, started_at=time.monotonic())

    @staticmethod
    def _pump(
        pipe, stream: str, loop: asyncio.AbstractEventLoop, on_output: Callable[[str, str], None]
    ) -> None:
        """后台守护线程:阻塞读取 Popen 管道,逐行经 call_soon_threadsafe 回到事件循环。"""
        if pipe is None:
            return
        try:
            for line in iter(pipe.readline, ""):
                line = line.rstrip("\r\n")
                if line:
                    loop.call_soon_threadsafe(on_output, line, stream)
        finally:
            try:
                pipe.close()
            except Exception:  # noqa: BLE001, S110
                pass

    async def _wait(self, pid: int) -> None:
        popen = self._procs.get(pid)
        if popen is None:
            self._wait_tasks.pop(pid, None)
            return
        rc = await asyncio.to_thread(popen.wait)
        cb = self._exit_cbs.get(pid)
        if cb:
            try:
                cb(rc if rc is not None else -1)
            except Exception:  # noqa: BLE001, S110
                pass
        self._procs.pop(pid, None)
        self._exit_cbs.pop(pid, None)
        self._readers.pop(pid, None)
        self._wait_tasks.pop(pid, None)

    def on_exit(self, pid: int, cb: Callable[[int], None]) -> None:
        self._exit_cbs[pid] = cb

    def alive(self, pid: int) -> bool:
        try:
            p = psutil.Process(pid)
            return p.status() != psutil.STATUS_ZOMBIE and p.is_running()
        except psutil.NoSuchProcess:
            return False
        except Exception:  # noqa: BLE001
            return False

    async def kill_tree(self, pid: int) -> bool:
        try:
            try:
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                for c in children:
                    try:
                        c.kill()
                    except psutil.NoSuchProcess:
                        pass
                parent.kill()
                _, alive = psutil.wait_procs([parent] + children, timeout=3)
                if not alive:
                    return True
            except psutil.NoSuchProcess:
                return True
            except Exception:  # noqa: BLE001, S110
                pass
            if os.name == "nt":
                try:
                    r = await asyncio.to_thread(
                        subprocess.run,
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                        timeout=5,
                    )
                    return r.returncode in (0, 128)
                except Exception:  # noqa: BLE001
                    return False
            else:
                try:
                    os.killpg(pid, signal.SIGKILL)
                    return True
                except ProcessLookupError:
                    return True
                except Exception:  # noqa: BLE001
                    return False
        finally:
            self._procs.pop(pid, None)
            self._exit_cbs.pop(pid, None)
            self._readers.pop(pid, None)

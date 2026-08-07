"""NVIDIA 适配器:nvidia-smi 查询 → DeviceInfo。"""

from __future__ import annotations

import shutil
import subprocess
from typing import NamedTuple

from . import DeviceInfo


class _GpuRow(NamedTuple):
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    util_pct: float
    temp_c: float | None
    freq_mhz: float | None


def _parse_smi(stdout: str) -> list[_GpuRow]:
    rows: list[_GpuRow] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            name = parts[0]
            total = int(parts[1])
            used = int(parts[2])
            free = int(parts[3])
            util = float(parts[4])
            temp = float(parts[5]) if parts[5] else None
            try:
                freq = float(parts[6]) if len(parts) > 6 and parts[6] else None
            except ValueError:  # clocks.gr 输出 "N/A" 等 → 频率 None,不丢行
                freq = None
            rows.append(_GpuRow(name, total, used, free, util, temp, freq))
        except (ValueError, IndexError):
            continue
    return rows


def _run_smi() -> str:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return ""
    try:
        r = subprocess.run(
            [
                smi,
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,clocks.gr",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return r.stdout if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001 — nvidia-smi 子进程异常/超时 → 视作无 NVIDIA,返回空
        return ""


class NvidiaAdapter:
    """nvidia-smi → DeviceInfo(device_name=产品原始名)。无 nvidia-smi / 无 NVIDIA → []。"""

    def enumerate(self) -> list[DeviceInfo]:
        return [
            DeviceInfo(
                row.name,
                "GPU",
                "VRAM",
                row.total_mb,
                row.free_mb,
                row.used_mb,
                row.util_pct,
                row.temp_c,
                row.freq_mhz,
            )
            for row in _parse_smi(_run_smi())
        ]

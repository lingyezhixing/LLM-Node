"""Intel GPU 适配器:Linux(i915 + intel_gpu_top)/Windows(LHM)统一接口。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from . import DeviceInfo
from .common import (
    _DRM_CLASS,
    _aggregate_sensors,
    _drm_cards,
    _lhm_computer,
    _system_mem,
)

_INTEL_IGPU_NAMES = {
    "8086:46d0": "Intel UHD Graphics (Alder Lake-N)",
    "8086:46d1": "Intel UHD Graphics (Alder Lake-N)",
}


def _intel_gpu_name(dev: Path) -> str:
    try:
        for line in (
            dev.joinpath("uevent").read_text(encoding="ascii", errors="ignore").splitlines()
        ):
            if line.startswith("PCI_ID="):
                pci_id = line.split("=", 1)[1].strip().lower()
                return _INTEL_IGPU_NAMES.get(pci_id, f"Intel UHD Graphics ({pci_id})")
    except OSError:
        pass
    return "Intel UHD Graphics"


def _is_i915(dev: Path) -> bool:
    try:
        uevent = dev.joinpath("uevent").read_text(encoding="ascii", errors="ignore")
    except OSError:
        return False
    return "DRIVER=i915" in uevent


def _run_intel_gpu_top() -> str | None:
    if shutil.which("intel_gpu_top") is None:
        return None
    try:
        r = subprocess.run(
            ["timeout", "2", "intel_gpu_top", "-J", "-s", "1000"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        return r.stdout if r.returncode in (0, 124) else None
    except Exception:  # noqa: BLE001
        return None


def _parse_intel_gpu_top(stdout: str | None) -> dict | None:
    if not stdout:
        return None
    import json

    decoder = json.JSONDecoder()
    buf, last = stdout, None
    while True:
        buf = buf.lstrip(" \t\r\n,[]")
        if not buf:
            break
        try:
            frame, end = decoder.raw_decode(buf)
        except json.JSONDecodeError:
            break
        buf = buf[end:]
        if not isinstance(frame, dict):
            break
        if frame.get("period", {}).get("duration", 0) >= 100:
            last = frame
    if last is None:
        return None
    busy = max((e.get("busy", 0.0) for e in last.get("engines", {}).values()), default=0.0)
    freq = last.get("frequency", {}).get("actual")
    power = last.get("power", {}).get("GPU")
    return {
        "busy_pct": busy,
        "freq_mhz": float(freq) if isinstance(freq, (int, float)) else None,
        "power_watts": float(power) if isinstance(power, (int, float)) else None,
    }


class IntelAdapter:
    """Intel GPU 适配器:Linux(i915 + intel_gpu_top)/Windows(LHM)统一接口。"""

    def enumerate(self) -> list[DeviceInfo]:
        if os.name == "nt":
            return self._enumerate_windows()
        return self._enumerate_linux()

    def _enumerate_linux(self) -> list[DeviceInfo]:
        if not _DRM_CLASS.is_dir():
            return []
        total, avail, used = _system_mem()
        cards = [c for c in _drm_cards() if _is_i915(c / "device")]
        if not cards:
            return []
        metrics = _parse_intel_gpu_top(_run_intel_gpu_top()) or {}
        return [
            DeviceInfo(
                _intel_gpu_name(c / "device"),
                "GPU (iGPU)",
                "Shared RAM",
                total,
                avail,
                used,
                metrics.get("busy_pct", 0.0),
                None,
                metrics.get("freq_mhz"),
                metrics.get("power_watts"),
            )
            for c in cards
        ]

    def _enumerate_windows(self) -> list[DeviceInfo]:
        c = _lhm_computer()
        if c is None:
            return []
        out: list[DeviceInfo] = []
        for hw in c.Hardware:
            if str(hw.HardwareType) != "GpuIntel":
                continue
            try:
                hw.Update()
                sensors = (
                    (str(s.SensorType), str(s.Name), s.Value if s.Value is not None else 0.0)
                    for s in hw.Sensors
                )
                out.append(_aggregate_sensors(str(hw.Name), sensors))
            except Exception:  # noqa: BLE001, S110
                pass
        return out

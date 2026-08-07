"""AMD GPU 适配器:Linux(amdgpu sysfs)/Windows(LHM)统一接口。"""

from __future__ import annotations

import os
from pathlib import Path

from . import DeviceInfo
from .common import (
    _DRM_CLASS,
    _aggregate_sensors,
    _drm_cards,
    _hwmon_temp1,
    _lhm_computer,
    _read_float,
    _read_int_mb,
)


def _is_amdgpu(dev: Path) -> bool:
    try:
        uevent = dev.joinpath("uevent").read_text(encoding="ascii", errors="ignore")
    except OSError:
        return False
    return "DRIVER=amdgpu" in uevent


def _amd_vram(dev: Path) -> tuple[int, int]:
    total = _read_int_mb(dev / "mem_info_vram_total")
    used = _read_int_mb(dev / "mem_info_vram_used")
    return total, used


_AMD_GPU_NAMES = {
    "1002:15fe": "AMD Radeon 780M Graphics",
}


def _amd_gpu_name(dev: Path) -> str:
    try:
        for line in (
            dev.joinpath("uevent").read_text(encoding="ascii", errors="ignore").splitlines()
        ):
            if line.startswith("PCI_ID="):
                pci_id = line.split("=", 1)[1].strip().lower()
                return _AMD_GPU_NAMES.get(pci_id, f"AMD Radeon ({pci_id})")
    except OSError:
        pass
    return "AMD Radeon"


class AmdAdapter:
    """AMD GPU 适配器:Linux(amdgpu sysfs)/Windows(LHM)统一接口。"""

    def enumerate(self) -> list[DeviceInfo]:
        if os.name == "nt":
            return self._enumerate_windows()
        return self._enumerate_linux()

    def _enumerate_linux(self) -> list[DeviceInfo]:
        if not _DRM_CLASS.is_dir():
            return []
        out: list[DeviceInfo] = []
        for card in _drm_cards():
            dev = card / "device"
            if not _is_amdgpu(dev):
                continue
            total, used = _amd_vram(dev)
            busy = _read_float(dev / "gpu_busy_percent")
            out.append(
                DeviceInfo(
                    _amd_gpu_name(dev),
                    "GPU (APU)",
                    "VRAM",
                    total,
                    max(total - used, 0),
                    used,
                    busy,
                    _hwmon_temp1(dev),
                )
            )
        return out

    def _enumerate_windows(self) -> list[DeviceInfo]:
        c = _lhm_computer()
        if c is None:
            return []
        out: list[DeviceInfo] = []
        for hw in c.Hardware:
            if str(hw.HardwareType) != "GpuAmd":
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

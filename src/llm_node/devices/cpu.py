"""主机 CPU 适配器:psutil RAM/占用 + 温度/频率(LHM 仅 Windows;hwmon/psutil 仅 Linux)。"""

from __future__ import annotations

import os

import psutil

from . import DeviceInfo
from .common import _lhm_computer, _system_mem


def _valid_reading(s) -> bool:
    """LHM 传感器读数佐证:Power/Clock/Temperature 值 > 0 才算有效。"""
    if str(getattr(s, "SensorType", "")) not in ("Power", "Clock", "Temperature"):
        return False
    v = getattr(s, "Value", None)
    if v is None:
        return False
    try:
        return float(v) > 0
    except (TypeError, ValueError):
        return False


def _lhm_cpu_temp() -> float | None:
    c = _lhm_computer()
    if c is None:
        return None
    try:
        for hw in c.Hardware:
            if str(hw.HardwareType) != "Cpu":
                continue
            hw.Update()
            zero_seen = False
            for s in hw.Sensors:
                if str(s.SensorType) != "Temperature" or not (
                    "Tctl" in str(s.Name) or "Tdie" in str(s.Name)
                ):
                    continue
                v = s.Value
                if v is None:
                    continue
                fv = float(v)
                if fv > 0:
                    return fv
                if fv == 0:
                    zero_seen = True
            if zero_seen and any(_valid_reading(x) for x in hw.Sensors):
                return 0.0
    except Exception:  # noqa: BLE001
        return None
    return None


def _cpu_temp_hwmon() -> float | None:
    sensors = getattr(psutil, "sensors_temperatures", None)
    if sensors is None:
        return None
    try:
        chips = sensors()
    except Exception:  # noqa: BLE001
        return None
    for chip in ("coretemp", "k10temp", "cpu_thermal"):
        for st in chips.get(chip, []):
            if st.current > 0 and st.label in ("Package id 0", "Tctl", "Tdie"):
                return float(st.current)
    for chip in ("coretemp", "k10temp", "cpu_thermal"):
        for st in chips.get(chip, []):
            if st.current > 0:
                return float(st.current)
    return None


def _cpu_temp() -> float | None:
    if os.name == "nt":
        return _lhm_cpu_temp()
    return _cpu_temp_hwmon()


def _lhm_cpu_freq() -> float | None:
    c = _lhm_computer()
    if c is None:
        return None
    try:
        best = 0.0
        for hw in c.Hardware:
            if str(hw.HardwareType) != "Cpu":
                continue
            hw.Update()
            for s in hw.Sensors:
                if str(s.SensorType) != "Clock" or "Core" not in str(s.Name):
                    continue
                v = float(s.Value) if s.Value is not None else None
                if v is not None and v > 0:
                    best = max(best, v)
        return float(best) if best > 0 else None
    except Exception:  # noqa: BLE001
        return None


def _cpu_freq_psutil() -> float | None:
    try:
        f = psutil.cpu_freq()
        return float(f.current) if f is not None and f.current > 0 else None
    except Exception:  # noqa: BLE001
        return None


def _cpu_freq() -> float | None:
    if os.name == "nt":
        return _lhm_cpu_freq()
    return _cpu_freq_psutil()


class CpuAdapter:
    """主机 CPU:psutil 取 RAM/占用 + _cpu_temp/_cpu_freq 取温度/频率。device_name='CPU'。"""

    def enumerate(self) -> list[DeviceInfo]:
        total, avail, used = _system_mem()
        try:
            usage = float(psutil.cpu_percent(interval=None))
        except Exception:  # noqa: BLE001
            return [DeviceInfo("CPU", "CPU", "RAM", total, avail, used, 0.0, None)]
        return [
            DeviceInfo("CPU", "CPU", "RAM", total, avail, used, usage, _cpu_temp(), _cpu_freq())
        ]

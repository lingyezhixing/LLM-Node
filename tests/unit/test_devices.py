from llm_node.devices.common import _aggregate_sensors
from llm_node.devices.nvidia import _parse_smi

# ==================== nvidia-smi 解析(_parse_smi)====================


def test_parse_smi_extracts_fields():
    out = "NVIDIA GeForce RTX 4060, 8192, 1024, 7168, 5, 45, 2475\n"
    rows = _parse_smi(out)
    assert len(rows) == 1
    r = rows[0]
    assert r.name == "NVIDIA GeForce RTX 4060"
    assert r.total_mb == 8192
    assert r.temp_c == 45.0
    assert r.freq_mhz == 2475.0


def test_parse_smi_skips_bad_lines():
    out = "good, 8192, 1024, 7168, 5, 45\nbroken line\n\n"
    assert len(_parse_smi(out)) == 1


def test_parse_smi_freq_na_keeps_row():
    out = "NVIDIA GeForce RTX 4060, 8192, 1024, 7168, 5, 45, N/A\n"
    rows = _parse_smi(out)
    assert len(rows) == 1
    assert rows[0].freq_mhz is None
    assert rows[0].temp_c == 45.0


# ==================== LHM 共享折叠(_aggregate_sensors)====================


def test_aggregate_sensors_dedicated_and_shared():
    sensors = [
        ("Load", "D3D", 42.0),
        ("SmallData", "Dedicated Used VRAM", 1000.0),
        ("SmallData", "Dedicated Total VRAM", 4000.0),
        ("SmallData", "Shared Used", 500.0),
        ("SmallData", "Shared Total", 2000.0),
        ("Clock", "GPU Core", 800.0),
        ("Clock", "GPU Memory", 2400.0),
        ("Temperature", "GPU Temp", 60.0),
    ]
    info = _aggregate_sensors("780M", sensors)
    assert info.device_name == "780M"
    assert info.device_type == "GPU (APU)"
    assert info.total_memory_mb == 6000
    assert info.used_memory_mb == 1500
    assert info.available_memory_mb == 4500
    assert info.temperature_celsius == 60.0
    assert info.freq_mhz == 800.0


# ==================== LHM 可用性(is_lhm_available)====================


def test_is_lhm_available_no_pythonnet(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "clr":
            raise ImportError("no pythonnet")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from llm_node.devices import common as cm

    assert cm.is_lhm_available() is False


def test_is_lhm_available_dll_present(monkeypatch, tmp_path):
    import sys
    import types

    from llm_node.devices import common as cm

    monkeypatch.setitem(sys.modules, "clr", types.ModuleType("clr"))
    fake_dll = tmp_path / "LibreHardwareMonitorLib.dll"
    fake_dll.write_text("fake")
    monkeypatch.setattr(cm, "_LHM_DLL", fake_dll)
    assert cm.is_lhm_available() is True


def test_is_lhm_available_dll_missing(monkeypatch, tmp_path):
    import sys
    import types

    from llm_node.devices import common as cm

    monkeypatch.setitem(sys.modules, "clr", types.ModuleType("clr"))
    monkeypatch.setattr(cm, "_LHM_DLL", tmp_path / "nonexistent.dll")
    assert cm.is_lhm_available() is False


# ==================== nvidia-smi 调用(_run_smi)====================


def test_run_smi_uses_noheader_nounits_format(monkeypatch):
    from llm_node.devices import nvidia as ad

    captured = {}

    class _R:
        returncode = 0
        stdout = "GPU, 8192, 0, 8192, 0, 40\n"

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _R()

    monkeypatch.setattr(ad.subprocess, "run", fake_run)
    ad._run_smi()
    assert "--format=csv,noheader,nounits" in captured["cmd"], captured["cmd"]


def test_parse_smi_handles_multi_gpu_csv_noheader_nounits():
    out = (
        "NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 1692, 6266, 35, 51\n"
        "Tesla V100-SXM2-32GB, 32768, 0, 32365, 0, 40\n"
    )
    rows = _parse_smi(out)
    assert len(rows) == 2
    assert "4060" in rows[0].name.lower() and rows[0].total_mb == 8188
    assert "v100" in rows[1].name.lower() and rows[1].total_mb == 32768


# ==================== token 匹配(_tokens / match_devices)====================


def test_tokens_splits_alnum():
    from llm_node.devices import _tokens

    assert _tokens("RTX 4060 Ti") == {"rtx", "4060", "ti"}
    assert _tokens("V100-SXM2") == {"v100", "sxm2"}
    assert _tokens("780M Graphics") == {"780m", "graphics"}
    assert _tokens("") == set()


def _di(name):
    from llm_node.devices import DeviceInfo

    return DeviceInfo(name, "GPU", "VRAM", 0, 0, 0, 0.0, None)


def test_match_devices_full_match_keyed_by_config_name():
    from llm_node.devices import match_devices

    candidates = [
        _di("NVIDIA GeForce RTX 4060 Ti"),
        _di("Tesla V100-SXM2-32GB"),
        _di("AMD Radeon 780M Graphics"),
        _di("CPU"),
    ]
    matched, unmatched = match_devices({"rtx 4060", "v100", "780m", "cpu"}, candidates)
    assert set(matched) == {"rtx 4060", "v100", "780m", "cpu"}
    assert matched["v100"].device_name == "Tesla V100-SXM2-32GB"
    assert unmatched == []


def test_match_devices_no_match_returns_empty_and_unmatched_preserved():
    from llm_node.devices import match_devices

    candidates = [_di("NVIDIA GeForce RTX 4060")]
    matched, unmatched = match_devices({"rtx 5090"}, candidates)
    assert matched == {}
    assert [c.device_name for c in unmatched] == ["NVIDIA GeForce RTX 4060"]


def test_match_devices_disambiguation_prefers_fewer_extra_tokens():
    from llm_node.devices import match_devices

    candidates = [_di("NVIDIA GeForce RTX 4060 Ti"), _di("NVIDIA GeForce RTX 4060")]
    matched, _ = match_devices({"rtx 4060"}, candidates)
    assert matched["rtx 4060"].device_name == "NVIDIA GeForce RTX 4060"


def test_match_devices_cpu_token_matches_cpu_candidate():
    from llm_node.devices import match_devices

    matched, unmatched = match_devices({"cpu"}, [_di("CPU")])
    assert "cpu" in matched
    assert unmatched == []


def test_match_devices_one_candidate_one_name():
    from llm_node.devices import match_devices

    matched, unmatched = match_devices({"4060", "rtx 4060"}, [_di("NVIDIA GeForce RTX 4060")])
    assert set(matched) == {"4060"}
    assert unmatched == []


def test_match_devices_requires_full_subset_not_partial():
    from llm_node.devices import match_devices

    matched, unmatched = match_devices({"rtx 4060"}, [_di("NVIDIA GeForce RTX 3090")])
    assert matched == {}
    assert [c.device_name for c in unmatched] == ["NVIDIA GeForce RTX 3090"]


# ==================== NvidiaAdapter====================


def test_enumerate_nvidia_returns_all_rows_with_raw_names(monkeypatch):
    from llm_node.devices import nvidia as ad

    smi = (
        "NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 1692, 6266, 35, 51\n"
        "Tesla V100-SXM2-32GB, 32768, 0, 32365, 0, 40\n"
    )
    monkeypatch.setattr(ad, "_run_smi", lambda: smi)
    out = ad.NvidiaAdapter().enumerate()
    assert len(out) == 2
    assert out[0].device_name == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert out[0].device_type == "GPU" and out[0].memory_type == "VRAM"
    assert out[0].total_memory_mb == 8188 and out[0].available_memory_mb == 6266
    assert out[1].device_name == "Tesla V100-SXM2-32GB" and out[1].total_memory_mb == 32768


def test_enumerate_nvidia_empty_when_no_smi(monkeypatch):
    from llm_node.devices import nvidia as ad

    monkeypatch.setattr(ad, "_run_smi", lambda: "")
    assert ad.NvidiaAdapter().enumerate() == []


# ==================== LHM 单例(_lhm_computer)====================


def test_lhm_computer_unavailable_returns_none(monkeypatch):
    from llm_node.devices import common as cm

    monkeypatch.setattr(cm, "is_lhm_available", lambda: False)
    assert cm._lhm_computer() is None


def test_lhm_computer_init_failure_returns_none(monkeypatch):
    import sys
    import types

    from llm_node.devices import common as cm

    def boom(*a, **k):
        raise RuntimeError("AddReference failed")

    fake_clr = types.ModuleType("clr")
    fake_clr.AddReference = boom
    monkeypatch.setattr(cm, "is_lhm_available", lambda: True)
    monkeypatch.setitem(sys.modules, "clr", fake_clr)
    monkeypatch.setattr(cm, "_LHM_COMPUTER", None)
    assert cm._lhm_computer() is None


# ==================== CPU 适配器(CpuAdapter / 温度 / 频率 / 平台分支)====================


def test_enumerate_cpu_basic(monkeypatch):
    from llm_node.devices import cpu as ad

    class _Mem:
        total = 16 * 1024**3
        available = 8 * 1024**3
        used = 8 * 1024**3

    monkeypatch.setattr(ad.psutil, "virtual_memory", lambda: _Mem())
    monkeypatch.setattr(ad.psutil, "cpu_percent", lambda interval=None: 33.0)
    monkeypatch.setattr(ad, "_cpu_temp", lambda: None)
    monkeypatch.setattr(ad, "_cpu_freq", lambda: 3700.0)
    out = ad.CpuAdapter().enumerate()
    assert len(out) == 1
    info = out[0]
    assert info.device_name == "CPU"
    assert info.device_type == "CPU" and info.memory_type == "RAM"
    assert info.total_memory_mb == 16 * 1024
    assert info.available_memory_mb == 8 * 1024
    assert info.usage_percentage == 33.0
    assert info.temperature_celsius is None
    assert info.freq_mhz == 3700.0


def test_enumerate_cpu_psutil_failure_degraded(monkeypatch):
    from llm_node.devices import cpu as ad

    def boom():
        raise OSError("psutil broke")

    monkeypatch.setattr(ad.psutil, "virtual_memory", boom)
    monkeypatch.setattr(ad, "_cpu_temp", lambda: None)
    out = ad.CpuAdapter().enumerate()
    assert len(out) == 1
    assert out[0].device_name == "CPU"
    assert out[0].total_memory_mb == 0


def test_lhm_cpu_temp_unavailable_returns_none(monkeypatch):
    from llm_node.devices import cpu as ad

    monkeypatch.setattr(ad, "_lhm_computer", lambda: None)
    assert ad._lhm_cpu_temp() is None


def _fake_cpu_hw(*sensors):
    import types

    class _FakeSensor:
        def __init__(self, stype, sname, val):
            self.SensorType = stype
            self.Name = sname
            self.Value = val

    class _FakeHardware:
        HardwareType = "Cpu"

        def __init__(self, sensors):
            self.Sensors = sensors

        def Update(self):
            pass

    return types.SimpleNamespace(Hardware=[_FakeHardware([_FakeSensor(*t) for t in sensors])])


def test_lhm_cpu_temp_reads_valid_tctl(monkeypatch):
    from llm_node.devices import cpu as ad

    monkeypatch.setattr(
        ad, "_lhm_computer", lambda: _fake_cpu_hw(("Temperature", "Core (Tctl/Tdie)", 78.125))
    )
    assert ad._lhm_cpu_temp() == 78.125


def test_lhm_cpu_temp_zero_without_corroboration_returns_none(monkeypatch):
    from llm_node.devices import cpu as ad

    monkeypatch.setattr(
        ad,
        "_lhm_computer",
        lambda: _fake_cpu_hw(
            ("Load", "CPU Core #1", 21.8),
            ("Load", "CPU Total", 20.3),
            ("Power", "Package", 0.0),
            ("Clock", "Core #1", float("nan")),
            ("Temperature", "Core (Tctl/Tdie)", 0.0),
        ),
    )
    assert ad._lhm_cpu_temp() is None


def test_lhm_cpu_temp_zero_with_corroboration_returns_zero(monkeypatch):
    from llm_node.devices import cpu as ad

    monkeypatch.setattr(
        ad,
        "_lhm_computer",
        lambda: _fake_cpu_hw(
            ("Power", "Package", 40.5),
            ("Clock", "Core #1", 4990.0),
            ("Temperature", "Core (Tctl/Tdie)", 0.0),
        ),
    )
    assert ad._lhm_cpu_temp() == 0.0


def test_lhm_cpu_temp_nan_never_returned(monkeypatch):
    from llm_node.devices import cpu as ad

    monkeypatch.setattr(
        ad,
        "_lhm_computer",
        lambda: _fake_cpu_hw(
            ("Power", "Package", 40.5),
            ("Temperature", "Core (Tctl/Tdie)", float("nan")),
        ),
    )
    assert ad._lhm_cpu_temp() is None


def test_lhm_cpu_temp_invalid_then_valid_second_sensor(monkeypatch):
    from llm_node.devices import cpu as ad

    monkeypatch.setattr(
        ad,
        "_lhm_computer",
        lambda: _fake_cpu_hw(
            ("Temperature", "Core (Tctl)", 0.0),
            ("Temperature", "Core (Tdie)", 55.0),
        ),
    )
    assert ad._lhm_cpu_temp() == 55.0


def test_cpu_temp_windows_branch_lhm(monkeypatch):
    from llm_node.devices import cpu as ad

    def boom():
        raise AssertionError("hwmon 不应被调用")

    monkeypatch.setattr(ad.os, "name", "nt")
    monkeypatch.setattr(ad, "_lhm_cpu_temp", lambda: 65.0)
    monkeypatch.setattr(ad, "_cpu_temp_hwmon", boom)
    assert ad._cpu_temp() == 65.0


def test_cpu_temp_linux_branch_hwmon(monkeypatch):
    import types

    from llm_node.devices import cpu as ad

    def boom():
        raise AssertionError("LHM 不应被调用")

    monkeypatch.setattr(ad.os, "name", "posix")
    monkeypatch.setattr(ad, "_lhm_cpu_temp", boom)
    monkeypatch.setattr(
        ad.psutil,
        "sensors_temperatures",
        lambda: {
            "coretemp": [types.SimpleNamespace(label="Package id 0", current=46.0)],
        },
        raising=False,
    )
    assert ad._cpu_temp() == 46.0


def test_cpu_temp_linux_branch_no_temp(monkeypatch):
    import types

    from llm_node.devices import cpu as ad

    monkeypatch.setattr(ad.os, "name", "posix")
    monkeypatch.setattr(
        ad.psutil,
        "sensors_temperatures",
        lambda: {
            "acpitz": [types.SimpleNamespace(label="", current=27.8)],
        },
        raising=False,
    )
    assert ad._cpu_temp() is None


def test_lhm_cpu_freq_reads_max_valid(monkeypatch):
    from llm_node.devices import cpu as ad

    monkeypatch.setattr(
        ad,
        "_lhm_computer",
        lambda: _fake_cpu_hw(
            ("Clock", "Core #1", 4990.0),
            ("Clock", "Core #2", 5000.0),
            ("Clock", "Core #3", float("nan")),
        ),
    )
    assert ad._lhm_cpu_freq() == 5000.0


def test_lhm_cpu_freq_all_invalid_returns_none(monkeypatch):
    from llm_node.devices import cpu as ad

    monkeypatch.setattr(
        ad,
        "_lhm_computer",
        lambda: _fake_cpu_hw(
            ("Clock", "Core #1", float("nan")),
            ("Clock", "Core #2", 0.0),
        ),
    )
    assert ad._lhm_cpu_freq() is None


def test_cpu_freq_psutil_reads_current(monkeypatch):
    import types

    from llm_node.devices import cpu as ad

    monkeypatch.setattr(
        ad.psutil, "cpu_freq", lambda: types.SimpleNamespace(current=3184.4, min=700.0, max=3400.0)
    )
    assert ad._cpu_freq_psutil() == 3184.4


def test_cpu_freq_psutil_unavailable_returns_none(monkeypatch):
    from llm_node.devices import cpu as ad

    def boom():
        raise OSError("no cpufreq")

    monkeypatch.setattr(ad.psutil, "cpu_freq", boom)
    assert ad._cpu_freq_psutil() is None
    monkeypatch.setattr(ad.psutil, "cpu_freq", lambda: None)
    assert ad._cpu_freq_psutil() is None


def test_cpu_freq_platform_branch(monkeypatch):
    from llm_node.devices import cpu as ad

    def boom():
        raise AssertionError("LHM 不应被调用")

    monkeypatch.setattr(ad.os, "name", "nt")
    monkeypatch.setattr(ad, "_lhm_cpu_freq", lambda: 4990.0)
    monkeypatch.setattr(ad, "_cpu_freq_psutil", boom)
    assert ad._cpu_freq() == 4990.0

    monkeypatch.setattr(ad.os, "name", "posix")
    monkeypatch.setattr(ad, "_cpu_freq_psutil", lambda: 3184.0)
    monkeypatch.setattr(ad, "_lhm_cpu_freq", boom)
    assert ad._cpu_freq() == 3184.0


def test_cpu_temp_hwmon_prefers_package_label(monkeypatch):
    import types

    from llm_node.devices import cpu as ad

    monkeypatch.setattr(
        ad.psutil,
        "sensors_temperatures",
        lambda: {
            "coretemp": [
                types.SimpleNamespace(label="Core 0", current=45.0),
                types.SimpleNamespace(label="Package id 0", current=47.0),
            ],
        },
        raising=False,
    )
    assert ad._cpu_temp_hwmon() == 47.0


def test_cpu_temp_hwmon_ignores_non_cpu_chips(monkeypatch):
    import types

    from llm_node.devices import cpu as ad

    monkeypatch.setattr(
        ad.psutil,
        "sensors_temperatures",
        lambda: {
            "acpitz": [types.SimpleNamespace(label="", current=27.8)],
            "it8613": [types.SimpleNamespace(label="", current=45.0)],
        },
        raising=False,
    )
    assert ad._cpu_temp_hwmon() is None


def test_cpu_temp_hwmon_unavailable_returns_none(monkeypatch):
    from llm_node.devices import cpu as ad

    def boom():
        raise OSError("no hwmon")

    monkeypatch.setattr(ad.psutil, "sensors_temperatures", boom, raising=False)
    assert ad._cpu_temp_hwmon() is None
    monkeypatch.setattr(ad.psutil, "sensors_temperatures", dict, raising=False)
    assert ad._cpu_temp_hwmon() is None


# ==================== DeviceMonitor / build_adapters====================


class _FakeAdapter:
    def __init__(self, devices):
        self._devices = devices

    def enumerate(self):
        return self._devices


def test_device_monitor_matches_config_names_and_keeps_unmatched():
    from llm_node.devices import DeviceInfo, DeviceMonitor

    def enum_gpus():
        return [
            DeviceInfo("NVIDIA GeForce RTX 4060", "GPU", "VRAM", 8188, 6266, 1692, 35.0, 51.0),
            DeviceInfo("Tesla V100-SXM2-32GB", "GPU", "VRAM", 32768, 32365, 0, 0.0, 40.0),
        ]

    def enum_cpu():
        return [DeviceInfo("CPU", "CPU", "RAM", 16384, 8192, 8192, 33.0, None)]

    mon = DeviceMonitor(
        [_FakeAdapter(enum_gpus()), _FakeAdapter(enum_cpu())], lambda: {"rtx 4060", "v100"}
    )
    mon.refresh()
    online = mon.online_devices()
    assert "rtx 4060" in online and "v100" in online
    assert "CPU" in online
    snap = mon.snapshot()
    assert snap["rtx 4060"].total_memory_mb == 8188
    assert snap["v100"].total_memory_mb == 32768


def test_device_monitor_dynamic_referenced_new_config_names_apply_without_restart():
    from llm_node.devices import DeviceInfo, DeviceMonitor

    def enum_gpus():
        return [
            DeviceInfo("NVIDIA GeForce RTX 4060", "GPU", "VRAM", 8188, 6266, 1692, 35.0, 51.0),
            DeviceInfo("NVIDIA GeForce GTX 1650", "GPU", "VRAM", 4096, 2000, 2096, 10.0, 45.0),
        ]

    referenced = {"rtx 4060"}
    mon = DeviceMonitor([_FakeAdapter(enum_gpus())], lambda: referenced)
    mon.refresh()
    assert "rtx 4060" in mon.online_devices()
    assert "NVIDIA GeForce RTX 4060" not in mon.online_devices()

    referenced = {"gtx 1650"}
    mon.refresh()
    online = mon.online_devices()
    assert "gtx 1650" in online
    assert "rtx 4060" not in online
    assert "NVIDIA GeForce RTX 4060" in online


def test_device_monitor_unmatched_referenced_is_offline():
    from llm_node.devices import DeviceMonitor

    mon = DeviceMonitor([_FakeAdapter([])], lambda: {"rtx 5090"})
    mon.refresh()
    assert "rtx 5090" not in mon.online_devices()


def test_device_monitor_enumerator_exception_isolated():
    from llm_node.devices import DeviceInfo, DeviceMonitor

    class _BoomAdapter:
        def enumerate(self):
            raise RuntimeError("backend broke")

    ok = DeviceInfo("CPU", "CPU", "RAM", 0, 0, 0, 0.0, None)
    mon = DeviceMonitor([_BoomAdapter(), _FakeAdapter([ok])], lambda: {"cpu"})
    mon.refresh()
    assert "cpu" in mon.online_devices()


def _ranked_adapter(cls_name, devices):
    cls = type(cls_name, (object,), {"enumerate": lambda self: devices})
    return cls()


def test_device_monitor_sorts_cpu_first_then_n_a_i():
    from llm_node.devices import DeviceInfo, DeviceMonitor

    nvidia = [
        DeviceInfo("NVIDIA GeForce RTX 4060", "GPU", "VRAM", 8188, 6266, 1692, 35.0, 51.0),
        DeviceInfo("NVIDIA GeForce RTX 4090", "GPU", "VRAM", 24564, 23540, 1024, 99.0, 70.0),
    ]
    amd = [
        DeviceInfo(
            "AMD Radeon 780M Graphics", "GPU (APU)", "Shared+Ded", 16394, 16376, 17, 0.0, 56.0
        )
    ]
    intel = [
        DeviceInfo(
            "Intel UHD Graphics (Alder Lake-N)",
            "GPU (iGPU)",
            "Shared RAM",
            15729,
            5866,
            9863,
            0.0,
            None,
        )
    ]
    cpu = [DeviceInfo("CPU", "CPU", "RAM", 16384, 8192, 8192, 33.0, None)]

    mon = DeviceMonitor(
        [
            _ranked_adapter("NvidiaAdapter", nvidia),
            _ranked_adapter("AmdAdapter", amd),
            _ranked_adapter("IntelAdapter", intel),
            _ranked_adapter("CpuAdapter", cpu),
        ],
        lambda: {"rtx 4060", "rtx 4090", "780m graphics", "alder lake-n", "cpu"},
    )
    mon.refresh()
    names = [d.device_name for d in mon.snapshot().values()]
    assert names == [
        "CPU",
        "NVIDIA GeForce RTX 4060",
        "NVIDIA GeForce RTX 4090",
        "AMD Radeon 780M Graphics",
        "Intel UHD Graphics (Alder Lake-N)",
    ]


def test_device_monitor_sorts_unmatched_too():
    from llm_node.devices import DeviceInfo, DeviceMonitor

    mon = DeviceMonitor(
        [
            _ranked_adapter(
                "NvidiaAdapter",
                [
                    DeviceInfo(
                        "NVIDIA GeForce RTX 4060", "GPU", "VRAM", 8188, 6266, 1692, 35.0, 51.0
                    )
                ],
            ),
            _ranked_adapter(
                "AmdAdapter",
                [
                    DeviceInfo(
                        "AMD Radeon 780M Graphics",
                        "GPU (APU)",
                        "Shared+Ded",
                        16394,
                        16376,
                        17,
                        0.0,
                        56.0,
                    )
                ],
            ),
            _ranked_adapter(
                "IntelAdapter",
                [
                    DeviceInfo(
                        "Intel UHD Graphics (Alder Lake-N)",
                        "GPU (iGPU)",
                        "Shared RAM",
                        15729,
                        5866,
                        9863,
                        0.0,
                        None,
                    )
                ],
            ),
            _ranked_adapter(
                "CpuAdapter", [DeviceInfo("CPU", "CPU", "RAM", 16384, 8192, 8192, 33.0, None)]
            ),
        ],
        lambda: set(),
    )
    mon.refresh()
    names = [d.device_name for d in mon.snapshot().values()]
    assert names == [
        "CPU",
        "NVIDIA GeForce RTX 4060",
        "AMD Radeon 780M Graphics",
        "Intel UHD Graphics (Alder Lake-N)",
    ]


def test_device_monitor_sorts_mixed_matched_and_unmatched():
    from llm_node.devices import DeviceInfo, DeviceMonitor

    nvidia = [DeviceInfo("NVIDIA GeForce RTX 4060", "GPU", "VRAM", 8188, 6266, 1692, 35.0, 51.0)]
    intel = [
        DeviceInfo(
            "Intel UHD Graphics (Alder Lake-N)",
            "GPU (iGPU)",
            "Shared RAM",
            15729,
            5866,
            9863,
            0.0,
            None,
        )
    ]
    cpu = [DeviceInfo("CPU", "CPU", "RAM", 16384, 8192, 8192, 33.0, None)]

    mon = DeviceMonitor(
        [
            _ranked_adapter("NvidiaAdapter", nvidia),
            _ranked_adapter("IntelAdapter", intel),
            _ranked_adapter("CpuAdapter", cpu),
        ],
        lambda: {"alder lake-n"},
    )
    mon.refresh()
    names = [d.device_name for d in mon.snapshot().values()]
    assert names == [
        "CPU",
        "NVIDIA GeForce RTX 4060",
        "Intel UHD Graphics (Alder Lake-N)",
    ]


def test_build_adapters_always_returns_four_adapters():
    from llm_node.devices import build_adapters

    ads = build_adapters()
    assert len(ads) == 4
    assert {type(a).__name__ for a in ads} == {
        "CpuAdapter",
        "NvidiaAdapter",
        "IntelAdapter",
        "AmdAdapter",
    }


# ==================== 集成(新模型引用实时匹配)====================


def test_new_gpu_model_matches_via_config_only(monkeypatch):
    import llm_node.devices as dev
    from llm_node.devices import nvidia as ad

    smi = "NVIDIA GeForce RTX 5090, 32768, 1000, 31768, 5, 45\n"
    monkeypatch.setattr(ad, "_run_smi", lambda: smi)
    matched, unmatched = dev.match_devices({"rtx 5090"}, ad.NvidiaAdapter().enumerate())
    assert "rtx 5090" in matched
    assert matched["rtx 5090"].device_name == "NVIDIA GeForce RTX 5090"
    assert matched["rtx 5090"].total_memory_mb == 32768
    assert unmatched == []


# ==================== Intel iGPU(i915 + intel_gpu_top)====================


def _make_i915_sysfs(tmp_path, pci_id="8086:46d1"):
    drm = tmp_path / "sys" / "class" / "drm"
    card0 = drm / "card0" / "device"
    card0.mkdir(parents=True)
    card0.joinpath("uevent").write_text(f"DRIVER=i915\nPCI_ID={pci_id}\n")
    card1 = drm / "card1" / "device"
    card1.mkdir(parents=True)
    card1.joinpath("uevent").write_text("DRIVER=amdgpu\nPCI_ID=1002:15fe\n")
    drm.joinpath("card0-DP-1").mkdir(parents=True)
    return drm


def test_intel_adapter_metrics_from_gpu_top(monkeypatch, tmp_path):
    from llm_node.devices import common as cm
    from llm_node.devices import intel as ad

    fake_drm = _make_i915_sysfs(tmp_path)
    monkeypatch.setattr(ad.os, "name", "posix")
    monkeypatch.setattr(ad, "_DRM_CLASS", fake_drm)
    monkeypatch.setattr(cm, "_DRM_CLASS", fake_drm)
    sample = (
        '[\n{"period": {"duration": 0.035, "unit": "ms"}, "engines": {"Render/3D": {"busy": 0.0}}}\n,'
        '{"period": {"duration": 1000.34, "unit": "ms"}, "frequency": {"actual": 2400.0, "requested": 2400.0},'
        ' "engines": {"Render/3D": {"busy": 12.5}, "Video": {"busy": 5.0}},'
        ' "power": {"GPU": 3.5, "Package": 2.4}}\n]'
    )
    monkeypatch.setattr(ad, "_run_intel_gpu_top", lambda: sample)
    monkeypatch.setattr(
        cm.psutil,
        "virtual_memory",
        lambda: type(
            "M", (), {"total": 16 * 1024**3, "available": 8 * 1024**3, "used": 8 * 1024**3}
        )(),
    )
    out = ad.IntelAdapter().enumerate()
    assert len(out) == 1
    info = out[0]
    assert info.device_name == "Intel UHD Graphics (Alder Lake-N)"
    assert info.device_type == "GPU (iGPU)" and info.memory_type == "Shared RAM"
    assert info.usage_percentage == 12.5
    assert info.freq_mhz == 2400.0
    assert info.power_watts == 3.5
    assert info.temperature_celsius is None


def test_intel_adapter_gpu_top_failure_degraded(monkeypatch, tmp_path):
    from llm_node.devices import common as cm
    from llm_node.devices import intel as ad

    fake_drm = _make_i915_sysfs(tmp_path)
    monkeypatch.setattr(ad.os, "name", "posix")
    monkeypatch.setattr(ad, "_DRM_CLASS", fake_drm)
    monkeypatch.setattr(cm, "_DRM_CLASS", fake_drm)
    monkeypatch.setattr(ad, "_run_intel_gpu_top", lambda: None)
    out = ad.IntelAdapter().enumerate()
    assert len(out) == 1
    assert out[0].usage_percentage == 0.0
    assert out[0].freq_mhz is None and out[0].power_watts is None


def test_intel_adapter_no_i915_returns_empty(monkeypatch, tmp_path):
    from llm_node.devices import common as cm
    from llm_node.devices import intel as ad

    monkeypatch.setattr(ad.os, "name", "posix")
    drm = tmp_path / "sys" / "class" / "drm"
    card1 = drm / "card1" / "device"
    card1.mkdir(parents=True)
    card1.joinpath("uevent").write_text("DRIVER=amdgpu\n")
    monkeypatch.setattr(ad, "_DRM_CLASS", drm)
    monkeypatch.setattr(cm, "_DRM_CLASS", drm)
    assert ad.IntelAdapter().enumerate() == []


def test_intel_adapter_windows_branch_via_lhm(monkeypatch):
    import types

    from llm_node.devices import intel as ad

    class _FakeSensor:
        def __init__(self, stype, sname, val):
            self.SensorType = stype
            self.Name = sname
            self.Value = val

    class _FakeHardware:
        def __init__(self, htype, name, sensors):
            self.HardwareType = htype
            self.Name = name
            self.Sensors = sensors

        def Update(self):
            pass

    fake_computer = types.SimpleNamespace(
        Hardware=[
            _FakeHardware(
                "GpuIntel",
                "Intel UHD Graphics",
                [
                    _FakeSensor("Load", "D3D", 42.0),
                    _FakeSensor("SmallData", "Dedicated Used VRAM", 1000.0),
                    _FakeSensor("SmallData", "Dedicated Total VRAM", 4000.0),
                    _FakeSensor("SmallData", "Shared Used", 500.0),
                    _FakeSensor("SmallData", "Shared Total", 2000.0),
                    _FakeSensor("Temperature", "GPU Temp", 60.0),
                ],
            ),
            _FakeHardware("GpuAmd", "AMD Radeon 780M", []),
            _FakeHardware("GpuNvidia", "NVIDIA GeForce RTX 4060", []),
        ]
    )
    monkeypatch.setattr(ad, "_lhm_computer", lambda: fake_computer)
    monkeypatch.setattr(ad.os, "name", "nt")
    out = ad.IntelAdapter().enumerate()
    assert len(out) == 1
    assert out[0].device_name == "Intel UHD Graphics"
    assert out[0].device_type == "GPU (APU)"


def test_intel_adapter_windows_lhm_unavailable_returns_empty(monkeypatch):
    from llm_node.devices import common as cm
    from llm_node.devices import intel as ad

    monkeypatch.setattr(cm, "is_lhm_available", lambda: False)
    monkeypatch.setattr(cm, "_LHM_COMPUTER", None)
    monkeypatch.setattr(ad.os, "name", "nt")
    assert ad.IntelAdapter().enumerate() == []


def test_intel_adapter_linux_missing_sysfs_returns_empty(monkeypatch, tmp_path):
    from llm_node.devices import common as cm
    from llm_node.devices import intel as ad

    monkeypatch.setattr(ad.os, "name", "posix")
    missing = tmp_path / "nonexistent"
    monkeypatch.setattr(ad, "_DRM_CLASS", missing)
    monkeypatch.setattr(cm, "_DRM_CLASS", missing)
    assert ad.IntelAdapter().enumerate() == []


def test_run_intel_gpu_top_timeout_124_still_returns_stdout(monkeypatch):
    from llm_node.devices import intel as ad

    class _R:
        returncode = 124
        stdout = '[\n{"period": {"duration": 1000.0, "unit": "ms"}}\n]'

    monkeypatch.setattr(ad.shutil, "which", lambda _: "/usr/bin/intel_gpu_top")
    monkeypatch.setattr(ad.subprocess, "run", lambda *a, **k: _R())
    assert ad._run_intel_gpu_top() == _R.stdout


def test_run_intel_gpu_top_real_failure_returns_none(monkeypatch):
    from llm_node.devices import intel as ad

    class _R:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(ad.shutil, "which", lambda _: "/usr/bin/intel_gpu_top")
    monkeypatch.setattr(ad.subprocess, "run", lambda *a, **k: _R())
    assert ad._run_intel_gpu_top() is None


def test_parse_intel_gpu_top_skips_init_frame_and_takes_last(monkeypatch):
    from llm_node.devices.intel import _parse_intel_gpu_top

    sample = (
        '[\n{"period": {"duration": 0.035, "unit": "ms"}, "engines": {"Render/3D": {"busy": 99.0}}}\n,'
        '{"period": {"duration": 1000.0, "unit": "ms"}, "frequency": {"actual": 1500.0},'
        ' "engines": {"Render/3D": {"busy": 10.0}, "Blitter": {"busy": 3.0}}, "power": {"GPU": 1.2}}\n,'
        '{"period": {"duration": 1000.0, "unit": "ms"}, "frequency": {"actual": 1800.0},'
        ' "engines": {"Render/3D": {"busy": 25.0}, "Video": {"busy": 5.0}}, "power": {"GPU": 2.2}}\n]'
    )
    m = _parse_intel_gpu_top(sample)
    assert m == {"busy_pct": 25.0, "freq_mhz": 1800.0, "power_watts": 2.2}


def test_parse_intel_gpu_top_pretty_multiline_real_format(monkeypatch):
    from llm_node.devices.intel import _parse_intel_gpu_top

    sample = (
        "[\n"
        "{\n"
        '\t"period": {\n'
        '\t\t"duration": 0.035112,\n'
        '\t\t"unit": "ms"\n'
        "\t},\n"
        '\t"engines": {\n'
        '\t\t"Render/3D": {\n'
        '\t\t\t"busy": 0.000000,\n'
        '\t\t\t"sema": 0.000000,\n'
        '\t\t\t"wait": 0.000000,\n'
        '\t\t\t"unit": "%"\n'
        "\t\t}\n"
        "\t}\n"
        "},\n"
        "{\n"
        '\t"period": {\n'
        '\t\t"duration": 1000.342406,\n'
        '\t\t"unit": "ms"\n'
        "\t},\n"
        '\t"frequency": {\n'
        '\t\t"requested": 2400.000000,\n'
        '\t\t"actual": 2400.000000,\n'
        '\t\t"unit": "MHz"\n'
        "\t},\n"
        '\t"engines": {\n'
        '\t\t"Render/3D": {\n'
        '\t\t\t"busy": 12.500000,\n'
        '\t\t\t"sema": 0.000000,\n'
        '\t\t\t"wait": 0.000000,\n'
        '\t\t\t"unit": "%"\n'
        "\t\t},\n"
        '\t\t"Video": {\n'
        '\t\t\t"busy": 5.000000,\n'
        '\t\t\t"sema": 0.000000,\n'
        '\t\t\t"wait": 0.000000,\n'
        '\t\t\t"unit": "%"\n'
        "\t\t}\n"
        "\t},\n"
        '\t"power": {\n'
        '\t\t"GPU": 3.500000,\n'
        '\t\t"Package": 2.389197,\n'
        '\t\t"unit": "W"\n'
        "\t}\n"
        "}\n"
        "]"
    )
    m = _parse_intel_gpu_top(sample)
    assert m == {"busy_pct": 12.5, "freq_mhz": 2400.0, "power_watts": 3.5}


def test_parse_intel_gpu_top_unparseable_returns_none():
    from llm_node.devices.intel import _parse_intel_gpu_top

    assert _parse_intel_gpu_top("") is None
    assert _parse_intel_gpu_top("garbage\nnot json") is None


# ==================== DeviceInfo 字段====================


def test_device_info_new_fields_default_none():
    from llm_node.devices import DeviceInfo

    d = DeviceInfo("X", "GPU", "VRAM", 1, 1, 0, 5.0, None)
    assert d.freq_mhz is None and d.power_watts is None


# ==================== AMD(amdgpu sysfs)====================


def _make_amdgpu_sysfs(
    tmp_path, busy="55", vram_total=8 * 1024**3, vram_used=3 * 1024**3, temp_mc=55000
):
    drm = tmp_path / "sys" / "class" / "drm"
    card0 = drm / "card0" / "device"
    card0.mkdir(parents=True)
    card0.joinpath("uevent").write_text("DRIVER=amdgpu\nPCI_ID=1002:15fe\n")
    card0.joinpath("gpu_busy_percent").write_text(busy)
    card0.joinpath("mem_info_vram_total").write_text(str(vram_total))
    card0.joinpath("mem_info_vram_used").write_text(str(vram_used))
    hwmon = card0 / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    hwmon.joinpath("temp1_input").write_text(str(temp_mc))
    return drm


def test_amd_adapter_basic(monkeypatch, tmp_path):
    from llm_node.devices import amd as ad
    from llm_node.devices import common as cm

    fake_drm = _make_amdgpu_sysfs(tmp_path)
    monkeypatch.setattr(ad.os, "name", "posix")
    monkeypatch.setattr(ad, "_DRM_CLASS", fake_drm)
    monkeypatch.setattr(cm, "_DRM_CLASS", fake_drm)
    out = ad.AmdAdapter().enumerate()
    assert len(out) == 1
    info = out[0]
    assert info.device_name == "AMD Radeon 780M Graphics"
    assert info.device_type == "GPU (APU)" and info.memory_type == "VRAM"
    assert info.usage_percentage == 55.0
    assert info.total_memory_mb == 8 * 1024 and info.used_memory_mb == 3 * 1024
    assert info.temperature_celsius == 55.0


def test_amd_adapter_missing_fields_degraded(monkeypatch, tmp_path):
    from llm_node.devices import amd as ad
    from llm_node.devices import common as cm

    monkeypatch.setattr(ad.os, "name", "posix")
    drm = tmp_path / "sys" / "class" / "drm"
    card0 = drm / "card0" / "device"
    card0.mkdir(parents=True)
    card0.joinpath("uevent").write_text("DRIVER=amdgpu\nPCI_ID=1002:1640\n")
    monkeypatch.setattr(ad, "_DRM_CLASS", drm)
    monkeypatch.setattr(cm, "_DRM_CLASS", drm)
    out = ad.AmdAdapter().enumerate()
    assert len(out) == 1
    assert out[0].usage_percentage == 0.0 and out[0].total_memory_mb == 0
    assert out[0].temperature_celsius is None
    assert out[0].device_name == "AMD Radeon (1002:1640)"


def test_amd_adapter_skips_non_amdgpu(monkeypatch, tmp_path):
    from llm_node.devices import amd as ad
    from llm_node.devices import common as cm

    fake_drm = _make_i915_sysfs(tmp_path)
    monkeypatch.setattr(ad.os, "name", "posix")
    monkeypatch.setattr(ad, "_DRM_CLASS", fake_drm)
    monkeypatch.setattr(cm, "_DRM_CLASS", fake_drm)
    out = ad.AmdAdapter().enumerate()
    assert len(out) == 1
    assert out[0].device_name == "AMD Radeon 780M Graphics"


def test_amd_adapter_available_clamped_non_negative(monkeypatch, tmp_path):
    from llm_node.devices import amd as ad
    from llm_node.devices import common as cm

    monkeypatch.setattr(ad.os, "name", "posix")
    fake_drm = _make_amdgpu_sysfs(tmp_path, vram_total=8 * 1024**3, vram_used=10 * 1024**3)
    monkeypatch.setattr(ad, "_DRM_CLASS", fake_drm)
    monkeypatch.setattr(cm, "_DRM_CLASS", fake_drm)
    out = ad.AmdAdapter().enumerate()
    assert out[0].total_memory_mb == 8 * 1024 and out[0].used_memory_mb == 10 * 1024
    assert out[0].available_memory_mb == 0


def test_amd_adapter_windows_branch_via_lhm(monkeypatch):
    import types

    from llm_node.devices import amd as ad

    class _FakeSensor:
        def __init__(self, stype, sname, val):
            self.SensorType = stype
            self.Name = sname
            self.Value = val

    class _FakeHardware:
        def __init__(self, htype, name, sensors):
            self.HardwareType = htype
            self.Name = name
            self.Sensors = sensors

        def Update(self):
            pass

    fake_computer = types.SimpleNamespace(
        Hardware=[
            _FakeHardware(
                "GpuAmd",
                "AMD Radeon 780M Graphics",
                [
                    _FakeSensor("Load", "D3D", 55.0),
                    _FakeSensor("SmallData", "Dedicated Used VRAM", 3000.0),
                    _FakeSensor("SmallData", "Dedicated Total VRAM", 8000.0),
                    _FakeSensor("SmallData", "Shared Used", 500.0),
                    _FakeSensor("SmallData", "Shared Total", 2000.0),
                    _FakeSensor("Temperature", "GPU Temp", 55.0),
                ],
            ),
            _FakeHardware("GpuIntel", "Intel UHD Graphics", []),
            _FakeHardware("GpuNvidia", "NVIDIA GeForce RTX 4060", []),
        ]
    )
    monkeypatch.setattr(ad, "_lhm_computer", lambda: fake_computer)
    monkeypatch.setattr(ad.os, "name", "nt")
    out = ad.AmdAdapter().enumerate()
    assert len(out) == 1
    assert out[0].device_name == "AMD Radeon 780M Graphics"
    assert out[0].device_type == "GPU (APU)"


def test_amd_adapter_windows_lhm_unavailable_returns_empty(monkeypatch):
    from llm_node.devices import amd as ad
    from llm_node.devices import common as cm

    monkeypatch.setattr(cm, "is_lhm_available", lambda: False)
    monkeypatch.setattr(cm, "_LHM_COMPUTER", None)
    monkeypatch.setattr(ad.os, "name", "nt")
    assert ad.AmdAdapter().enumerate() == []


def test_amd_adapter_linux_missing_sysfs_returns_empty(monkeypatch, tmp_path):
    from llm_node.devices import amd as ad
    from llm_node.devices import common as cm

    monkeypatch.setattr(ad.os, "name", "posix")
    missing = tmp_path / "nonexistent"
    monkeypatch.setattr(ad, "_DRM_CLASS", missing)
    monkeypatch.setattr(cm, "_DRM_CLASS", missing)
    assert ad.AmdAdapter().enumerate() == []

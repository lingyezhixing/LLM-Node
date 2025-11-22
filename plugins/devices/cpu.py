import psutil
from typing import Dict, Any
from plugins.devices.Base_Class import DevicePlugin
import logging

logger = logging.getLogger(__name__)

class CPUDevice(DevicePlugin):
    """CPU设备插件"""

    def __init__(self):
        super().__init__("CPU")

    def is_online(self) -> bool:
        """CPU通常总是在线的"""
        return True

    def get_devices_info(self) -> Dict[str, Any]:
        """获取CPU设备信息"""
        try:
            memory = psutil.virtual_memory()
            total_mb = memory.total // (1024 * 1024)
            available_mb = memory.available // (1024 * 1024)
            used_mb = memory.used // (1024 * 1024)
            usage_percentage = psutil.cpu_percent(interval=1)

            temperature = None
            try:
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        for name, entries in temps.items():
                            if entries:
                                temp = entries[0].current
                                if temp is not None:
                                    temperature = temp
                                    break
            except Exception:
                temperature = None

            device_info = {
                'device_type': 'CPU',
                'memory_type': 'RAM',
                'total_memory_mb': total_mb,
                'available_memory_mb': available_mb,
                'used_memory_mb': used_mb,
                'usage_percentage': usage_percentage,
                'temperature_celsius': temperature
            }

            logger.debug(f"CPU设备: {device_info}")
            return device_info

        except Exception as e:
            logger.error(f"获取CPU设备信息失败: {e}")
            return {
                'device_type': 'CPU',
                'memory_type': 'RAM',
                'total_memory_mb': 0,
                'available_memory_mb': 0,
                'used_memory_mb': 0,
                'usage_percentage': 0.0,
                'temperature_celsius': None
            }
import GPUtil
from typing import Dict, Any
from plugins.devices.Base_Class import DevicePlugin
import logging

logger = logging.getLogger(__name__)

class RTX4060Device(DevicePlugin):
    """RTX 4060设备插件"""

    def __init__(self):
        super().__init__("rtx 4060")

    def is_online(self) -> bool:
        """检查RTX 4060是否在线"""
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                # 检查GPU名称是否包含4060
                if "4060" in gpu.name:
                    logger.debug(f"检测到RTX 4060: {gpu.name}")
                    return True
            return False
        except Exception as e:
            logger.error(f"检查RTX 4060在线状态失败: {e}")
            return False

    def get_devices_info(self) -> Dict[str, Any]:
        """获取RTX 4060设备信息"""
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if "4060" in gpu.name:
                    total_mb = int(gpu.memoryTotal)
                    used_mb = int(gpu.memoryUsed)
                    available_mb = int(gpu.memoryFree)
                    usage_percentage = gpu.load * 100
                    temperature = gpu.temperature

                    device_info = {
                        'device_type': 'GPU',
                        'memory_type': 'VRAM',
                        'total_memory_mb': total_mb,
                        'available_memory_mb': available_mb,
                        'used_memory_mb': used_mb,
                        'usage_percentage': usage_percentage,
                        'temperature_celsius': temperature
                    }

                    logger.debug(f"RTX 4060设备: {device_info}")
                    return device_info

            logger.warning("未找到RTX 4060 GPU")
            return {
                'device_type': 'GPU',
                'memory_type': 'VRAM',
                'total_memory_mb': 0,
                'available_memory_mb': 0,
                'used_memory_mb': 0,
                'usage_percentage': 0.0,
                'temperature_celsius': None
            }

        except Exception as e:
            logger.error(f"获取RTX 4060设备信息失败: {e}")
            return {
                'device_type': 'GPU',
                'memory_type': 'VRAM',
                'total_memory_mb': 0,
                'available_memory_mb': 0,
                'used_memory_mb': 0,
                'usage_percentage': 0.0,
                'temperature_celsius': None
            }


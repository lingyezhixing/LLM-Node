from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DevicePlugin(ABC):
    """设备插件基类"""

    def __init__(self, device_name: str):
        self.device_name = device_name
        logger.debug(f"设备插件初始化: {device_name}")

    @abstractmethod
    def is_online(self) -> bool:
        """
        检查设备是否在线
        返回: bool - 设备是否在线
        """
        pass

    @abstractmethod
    def get_devices_info(self) -> Dict[str, Any]:
        """
        获取设备详细信息
        返回: {
            'device_type': str,  # 设备类型，由插件自定义
            'memory_type': str,  # 内存类型，由插件自定义
            'total_memory_mb': int,  # 总内存MB
            'available_memory_mb': int,  # 可用内存MB
            'used_memory_mb': int,  # 已用内存MB
            'usage_percentage': float,  # 使用率百分比
            'temperature_celsius': float | None,  # 设备温度（摄氏度），不可用时为None
        }
        """
        pass
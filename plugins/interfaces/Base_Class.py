from abc import ABC, abstractmethod
from typing import Tuple, Set
import logging

logger = logging.getLogger(__name__)

class InterfacePlugin(ABC):
    """接口插件基类"""

    def __init__(self, interface_name: str, model_manager=None):
        self.interface_name = interface_name
        self.model_manager = model_manager
        logger.debug(f"接口插件初始化: {interface_name}")

    @abstractmethod
    def health_check(self, model_alias: str, port: int, start_time: float = None, timeout_seconds: int = 300) -> Tuple[bool, str]:
        """
        对指定模型进行健康检查
        参数:
            model_alias: 模型别名
            port: 模型服务端口
            start_time: 检查开始时间
            timeout_seconds: 超时时间
        返回: (是否健康, 健康状态描述)
        """
        pass

    @abstractmethod
    def get_supported_endpoints(self) -> Set[str]:
        """
        获取该接口支持的API端点
        返回: 支持的端点路径集合，如 {"v1/chat/completions", "v1/completions"}
        """
        pass

    @abstractmethod
    def validate_request(self, path: str, model_alias: str) -> Tuple[bool, str]:
        """
        验证请求路径是否适合该接口类型
        参数:
            path: 请求路径
            model_alias: 模型别名
        返回: (是否有效, 错误消息)
        """
        pass

    
import openai
import time
from typing import Tuple, Set
import logging
from plugins.interfaces.base import InterfacePlugin

logger = logging.getLogger(__name__)

class ChatInterface(InterfacePlugin):
    """聊天接口插件"""

    def __init__(self, model_manager=None):
        super().__init__("Chat", model_manager)
        self.async_clients = {}


    def health_check(self, model_alias: str, port: int, start_time: float = None, timeout_seconds: int = 300) -> Tuple[bool, str]:
        """聊天模型健康检查 - 先浅层检查，再深层检查"""
        if start_time is None:
            start_time = time.time()

        # 第一阶段：浅层检查 - 验证服务是否可用
        logger.debug(f"聊天接口开始浅层检查: {model_alias}:{port}")
        while time.time() - start_time < timeout_seconds:
            try:
                client = openai.OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="dummy-key")
                client.models.list(timeout=3.0)
                logger.debug(f"聊天接口浅层检查通过: {model_alias}:{port}")
                break
            except Exception as e:
                logger.debug(f"聊天接口浅层检查失败: {e}")
                time.sleep(2)
        else:
            return False, f"聊天接口浅层检查超时: 服务在 {timeout_seconds} 秒内不可用"

        # 第二阶段：深层检查 - 验证聊天接口功能
        logger.debug(f"聊天接口开始深层检查: {model_alias}:{port}")
        while time.time() - start_time < timeout_seconds:
            try:
                client = openai.OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="dummy-key")
                client.chat.completions.create(
                    model=model_alias,
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=1,
                    stream=False,
                    timeout=5.0
                )
                logger.debug(f"聊天接口深层检查通过: {model_alias}:{port}")
                return True, "聊天接口健康检查成功"
            except openai.APIConnectionError as e:
                logger.debug(f"聊天接口深层检查API连接错误: {e.__cause__}")
            except openai.APIStatusError as e:
                logger.debug(f"聊天接口深层检查返回非成功状态码: {e.status_code} - {e.response}")
            except openai.APITimeoutError:
                logger.debug(f"聊天接口深层检查请求超时")
            except Exception as e:
                logger.warning(f"聊天接口深层检查期间出现意外错误: {e}")

            time.sleep(1)

        return False, "聊天接口深层检查超时"

    def get_supported_endpoints(self) -> Set[str]:
        """获取聊天接口支持的API端点"""
        return {"v1/chat/completions"}

    def validate_request(self, path: str, model_alias: str) -> Tuple[bool, str]:
        """验证请求路径是否适合聊天接口"""
        is_completion_endpoint = "v1/completions" in path

        if is_completion_endpoint:
            return False, f"模型 '{model_alias}' 是 'Chat' 模式, 不支持文本补全接口"

        return True, ""

    
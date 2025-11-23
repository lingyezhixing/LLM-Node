#!/usr/bin/env python3
"""
LLM-Node 计算节点主程序
纯后端版本：无GUI，无数据库，无Token统计
"""

import sys
import os
import concurrent.futures
from utils.logger import setup_logging, get_logger
from core.config_manager import ConfigManager
from core.api_server import run_api_server
from core.process_manager import get_process_manager, cleanup_process_manager
from core.model_controller import ModelController

# 修改默认配置文件路径
CONFIG_PATH = 'config.yaml'

class NodeApplication:
    """LLM 计算节点应用程序"""

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self.config_manager = None
        self.model_controller = None
        self.logger = None
        self.running = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def setup_logging(self) -> None:
        """设置日志系统"""
        if self.config_manager:
            log_level = self.config_manager.get_log_level()
        else:
            log_level = os.environ.get('LOG_LEVEL', 'INFO')
        setup_logging(log_level=log_level)
        self.logger = get_logger(__name__)

    def initialize(self) -> None:
        """初始化节点组件"""
        # 1. 加载配置
        if not os.path.exists(self.config_path):
            print(f"配置文件不存在: {self.config_path}")
            sys.exit(1)
            
        self.config_manager = ConfigManager(self.config_path)
        self.setup_logging()
        self.logger.info(">>> 正在启动 LLM 计算节点 (Node Mode) <<<")

        # 2. 初始化进程管理器
        get_process_manager()
        self.logger.info("进程管理器已就绪")

        # 3. 初始化模型控制器 (无数据库依赖)
        self.model_controller = ModelController(self.config_manager)
        self.logger.info("模型控制器已就绪")

    def start(self) -> None:
        """启动服务"""
        try:
            self.initialize()
            self.running = True

            # 启动 API 服务器 (阻塞运行)
            self.logger.info("正在启动节点 API 服务...")
            run_api_server(self.config_manager, self.model_controller)

        except KeyboardInterrupt:
            self.logger.info("接收到停止信号...")
        except Exception as e:
            self.logger.error(f"节点启动失败: {e}", exc_info=True)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """关闭节点"""
        if not self.running:
            return
            
        self.logger.info("正在关闭节点服务...")
        self.running = False
        
        if self.model_controller:
            self.model_controller.shutdown()
            
        cleanup_process_manager()
        self.executor.shutdown(wait=True)
        self.logger.info("节点服务已安全关闭")

if __name__ == "__main__":
    app = NodeApplication()
    app.start()
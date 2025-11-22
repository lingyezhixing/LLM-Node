import logging
import os
import glob
import sys
from datetime import datetime
from typing import Optional

# 全局变量，用于保存 LogManager 实例，以便后续修改级别或获取文件路径
_log_manager_instance = None

class LogManager:
    """日志管理器：负责配置日志文件、清理旧日志和挂载处理器"""
    def __init__(self, log_level: str = "INFO", log_dir: str = "logs"):
        self.log_level = log_level
        self.log_dir = log_dir
        self.current_log_file = None

        # 1. 确保日志目录存在
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except Exception as e:
                print(f"[Logger] 创建日志目录失败: {e}")

        # 2. 清理旧日志
        self._cleanup_old_logs()

        # 3. 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.current_log_file = os.path.join(log_dir, f"LLM-Manager_{timestamp}.log")

        # 4. 配置根日志器（核心逻辑）
        self._configure_root_logger()

    def _cleanup_old_logs(self):
        """清理旧日志文件，保留最新的10个"""
        try:
            log_pattern = os.path.join(self.log_dir, "LLM-Manager_*.log")
            log_files = glob.glob(log_pattern)
            # 按修改时间排序，保留最新的10个
            log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            for log_file in log_files[10:]:
                try:
                    os.remove(log_file)
                except Exception:
                    pass
        except Exception as e:
            print(f"[Logger] 清理旧日志失败: {e}")

    def _configure_root_logger(self):
        """配置根日志器，挂载文件和控制台处理器"""
        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        # 获取根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # --- 清理旧的处理器 (关键步骤) ---
        # 防止重复添加 Handler 导致日志重复打印
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        # 1. 添加控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # 2. 添加文件处理器 (只有在 setup_logging 被调用时才会发生)
        try:
            file_handler = logging.FileHandler(self.current_log_file, encoding='utf-8')
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            print(f"[Logger] 日志系统已初始化，文件: {self.current_log_file}")
        except Exception as e:
            print(f"[Logger] 无法创建日志文件处理器: {e}")

    def set_level(self, level: str):
        """动态修改日志级别"""
        self.log_level = level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)

# --- 模块级函数 ---

def get_logger(name: str) -> logging.Logger:
    """
    获取日志器
    注意：此函数现在是轻量级的，不会触发文件创建或系统配置。
    """
    # 如果还没有初始化，Python logging 会默认使用 Warning 级别且只输出到 stderr
    # 为了在初始化前也能看到 INFO 日志，我们在这里做一个极简的控制台配置
    # 但绝不创建文件
    if len(logging.getLogger().handlers) == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    return logging.getLogger(name)

def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """
    初始化日志系统
    这是创建日志文件的唯一入口。
    """
    global _log_manager_instance
    
    # 如果已经初始化过，只更新级别，不再创建新文件
    if _log_manager_instance is not None:
        _log_manager_instance.set_level(log_level)
        return

    # 只有这里才会真正实例化 LogManager，从而创建文件
    _log_manager_instance = LogManager(log_level, log_dir)
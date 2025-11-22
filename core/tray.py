import logging
import webbrowser
import os
import time
import requests
from PIL import Image
from pystray import Icon as TrayIcon, Menu as TrayMenu, MenuItem as TrayMenuItem
from typing import Optional
from utils.logger import get_logger
from core.config_manager import ConfigManager

logger = get_logger(__name__)

class SystemTray:
    """系统托盘服务"""

    def __init__(self, config_manager: ConfigManager):
        # 接收配置管理器实例
        self.config_manager = config_manager

        # 从配置管理器获取服务器配置
        server_cfg = self.config_manager.get_openai_config()
        self.server_host = server_cfg['host'] if server_cfg['host'] != '0.0.0.0' else 'localhost'
        self.server_port = server_cfg['port']

        self.server_url = f"http://{self.server_host}:{self.server_port}"
        self.tray_icon: Optional[TrayIcon] = None
        self.exit_callback = None

        logger.info(f"托盘服务初始化完成，连接到API: {self.server_url}")

    def open_webui(self):
        """打开WebUI"""
        logger.info("正在打开WebUI...")
        try:
            webbrowser.open(self.server_url)
            logger.info(f"已在浏览器中打开WebUI: {self.server_url}")
        except Exception as e:
            logger.error(f"打开WebUI失败: {e}")
            logger.info("请手动访问WebUI地址")

    def restart_auto_start_models(self):
        """重启所有auto_start模型"""
        logger.info("正在执行指令：重启所有 'auto_start' 模型...")
        try:
            response = requests.post(f"{self.server_url}/api/models/restart-autostart", timeout=30)
            result = response.json()
            if result.get("success"):
                logger.info(f"成功重启autostart模型: {result.get('started_models', [])}")
            else:
                logger.error(f"重启autostart模型失败: {result.get('message', '未知错误')}")
        except Exception as e:
            logger.error(f"通过API重启autostart模型失败: {e}")

    def unload_all_models(self):
        """卸载全部模型"""
        logger.info("正在执行指令：卸载全部模型...")
        try:
            response = requests.post(f"{self.server_url}/api/models/stop-all", timeout=30)
            result = response.json()
            if result.get("success"):
                logger.info("全部模型卸载完毕。")
            else:
                logger.error(f"卸载全部模型失败: {result.get('message', '未知错误')}")
        except Exception as e:
            logger.error(f"通过API卸载全部模型失败: {e}")

    def get_tray_title(self) -> str:
        """获取托盘标题"""
        try:
            response = requests.get(f"{self.server_url}/api/devices/info", timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    devices = result.get("devices", {})
                    online_devices = [
                        name for name, info in devices.items()
                        if info.get("online", False)
                    ]

                    if online_devices:
                        return f"LLM-Manager (设备: {', '.join(online_devices)})"
        except Exception as e:
            logger.debug(f"获取设备信息失败: {e}")

        return "LLM-Manager (设备状态未知)"

    def exit_application(self):
        """退出应用程序 - 通过API服务器关闭所有模型"""
        logger.info("正在退出应用程序...")

        # 通过API服务器关闭所有模型
        try:
            logger.info("正在通过API关闭所有模型...")
            response = requests.post(f"{self.server_url}/api/models/stop-all", timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    logger.info("所有模型已通过API关闭")
                else:
                    logger.warning("通过API关闭模型失败")
            else:
                logger.warning("无法连接到API服务器关闭模型")
        except Exception as e:
            logger.warning(f"通过API关闭模型时出错: {e}")

        # 调用退出回调
        if self.exit_callback:
            logger.info("调用退出回调...")
            try:
                self.exit_callback()
            except Exception as e:
                logger.error(f"退出回调执行失败: {e}")

        logger.info("程序退出")
        os._exit(0)

    def start_tray(self):
        """创建并运行系统托盘图标"""
        try:
            icon_path = os.path.join(os.path.dirname(__file__), '..', 'icons', 'icon.ico')
            if not os.path.exists(icon_path):
                logger.error(f"图标文件未找到: {icon_path}。将使用默认图标。")
                image = Image.new('RGB', (64, 64), 'black')
            else:
                image = Image.open(icon_path)

            menu = TrayMenu(
                TrayMenuItem('打开 WebUI', self.open_webui, default=True),
                TrayMenu.SEPARATOR,
                TrayMenuItem('重启 Auto-Start 模型', self.restart_auto_start_models),
                TrayMenuItem('卸载全部模型', self.unload_all_models),
                TrayMenu.SEPARATOR,
                TrayMenuItem('退出', self.exit_application)
            )

            self.tray_icon = TrayIcon(
                "LLM-Manager",
                image,
                self.get_tray_title(),
                menu
            )

            logger.info("系统托盘图标已创建。")
            self.tray_icon.run()

        except Exception as e:
            logger.error(f"创建系统托盘图标失败: {e}")
            self.exit_application()

    def set_exit_callback(self, callback):
        """设置退出回调函数"""
        self.exit_callback = callback
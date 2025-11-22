"""
配置管理器 - 节点版
"""
import json
import threading
import os
from typing import Dict, List, Optional, Any, Set
from utils.logger import get_logger

logger = get_logger(__name__)

class ConfigManager:
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.config = {}
        self.alias_to_primary_name = {}
        self.config_lock = threading.Lock()
        self.load_config()

    def load_config(self):
        with self.config_lock:
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self._init_alias_mapping()
            except Exception as e:
                logger.error(f"加载配置失败: {e}")
                raise

    def _init_alias_mapping(self):
        self.alias_to_primary_name.clear()
        for key, cfg in self.config.items():
            if key == "program": continue
            primary = cfg.get("aliases", [key])[0]
            for alias in cfg.get("aliases", []):
                self.alias_to_primary_name[alias] = primary

    def resolve_primary_name(self, alias: str) -> str:
        return self.alias_to_primary_name.get(alias, alias)

    def get_program_config(self):
        return self.config.get("program", {})

    def get_model_config(self, name):
        primary = self.resolve_primary_name(name)
        # 简单遍历查找，因为 key 不一定是 primary name
        for key, cfg in self.config.items():
            if key == "program": continue
            if cfg.get("aliases", []) and cfg["aliases"][0] == primary:
                return cfg
        return None

    def get_model_names(self):
        return [cfg["aliases"][0] for key, cfg in self.config.items() 
                if key != "program" and "aliases" in cfg]

    def get_adaptive_model_config(self, alias: str, online_devices: Set[str]):
        """获取适配当前硬件的模型启动配置"""
        base_config = self.get_model_config(alias)
        if not base_config: return None
        
        # 优先查找具体硬件配置块
        for key, val in base_config.items():
            if isinstance(val, dict) and "required_devices" in val:
                req = set(val["required_devices"])
                if req.issubset(online_devices):
                    # 构造运行配置
                    run_cfg = base_config.copy()
                    # 清理顶层非通用配置
                    for k in list(run_cfg.keys()):
                        if k not in ["aliases", "mode", "port", "auto_start"]:
                            del run_cfg[k]
                    
                    run_cfg.update(val)
                    run_cfg["config_source"] = key
                    return run_cfg
        return None

    # --- Getters ---
    def get_openai_config(self):
        return {
            "host": self.get_program_config().get('host', '0.0.0.0'),
            "port": self.get_program_config().get('port', 8080)
        }

    def get_device_plugin_dir(self):
        return self.get_program_config().get('device_plugin_dir', 'plugins/devices')

    def get_interface_plugin_dir(self):
        return self.get_program_config().get('interface_plugin_dir', 'plugins/interfaces')
    
    def get_alive_time(self):
        return self.get_program_config().get('alive_time', 60)

    def get_log_level(self):
        return self.get_program_config().get('log_level', 'INFO')

    def is_gpu_monitoring_disabled(self):
        return self.get_program_config().get('Disable_GPU_monitoring', False)
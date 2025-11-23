"""
配置管理器 - 节点版 (纯 YAML 版)
"""
import yaml
import threading
import os
from typing import List, Set
from utils.logger import get_logger

logger = get_logger(__name__)

class ConfigManager:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = {}
        self.alias_to_primary_name = {}
        self.config_lock = threading.Lock()
        self.load_config()

    def load_config(self):
        with self.config_lock:
            try:
                if not os.path.exists(self.config_path):
                    raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

                with open(self.config_path, 'r', encoding='utf-8') as f:
                    # 使用 yaml.safe_load 解析
                    self.config = yaml.safe_load(f) or {}

                self._init_alias_mapping()
                logger.info(f"成功加载配置: {self.config_path}")
            except Exception as e:
                logger.error(f"加载配置失败: {e}")
                raise

    def _init_alias_mapping(self):
        self.alias_to_primary_name.clear()
        for key, cfg in self.config.items():
            if key == "program": continue
            
            aliases = cfg.get("aliases", [key])
            # 确保 aliases 是列表且不为空
            if not aliases: aliases = [key]
            
            primary = aliases[0]
            for alias in aliases:
                self.alias_to_primary_name[alias] = primary

    def resolve_primary_name(self, alias: str) -> str:
        return self.alias_to_primary_name.get(alias, alias)

    def get_program_config(self):
        return self.config.get("program", {})

    def get_model_config(self, name):
        primary = self.resolve_primary_name(name)
        for key, cfg in self.config.items():
            if key == "program": continue
            aliases = cfg.get("aliases", [key])
            if aliases and aliases[0] == primary:
                return cfg
        return None

    def get_model_names(self):
        return [cfg.get("aliases", [key])[0] for key, cfg in self.config.items() 
                if key != "program"]

    def get_adaptive_model_config(self, alias: str, online_devices: Set[str]):
        """获取适配当前硬件的模型启动配置"""
        base_config = self.get_model_config(alias)
        if not base_config: return None
        
        # 优先查找具体硬件配置块
        for key, val in base_config.items():
            # 识别是否为硬件配置块（必须包含 required_devices）
            if isinstance(val, dict) and "required_devices" in val:
                req = set(val["required_devices"])
                if req.issubset(online_devices):
                    # 构造运行配置
                    run_cfg = base_config.copy()
                    # 清理顶层非通用配置
                    for k in list(run_cfg.keys()):
                        if k not in ["aliases", "mode", "port", "auto_start"]:
                            del run_cfg[k]
                    
                    # 更新特定硬件的配置
                    run_cfg.update({
                        "script_path": val["script_path"], 
                        "memory_mb": val["memory_mb"],
                        "required_devices": val.get("required_devices", []),
                        "config_source": key
                    })
                    return run_cfg
        return None

    def validate_config(self) -> List[str]:
        """验证配置文件的有效性"""
        errors = []
        try:
            for key, model_cfg in self.config.items():
                if key == "program": continue

                has_device_config = False
                for cfg_key in model_cfg.keys():
                    if cfg_key not in ["aliases", "mode", "port", "auto_start"]:
                        device_config = model_cfg[cfg_key]
                        if isinstance(device_config, dict):
                            has_device_config = True
                            required_device_keys = ['required_devices', 'script_path', 'memory_mb']
                            for req_key in required_device_keys:
                                if req_key not in device_config:
                                    errors.append(f"模型 '{key}' 的设备配置 '{cfg_key}' 缺少必需项: {req_key}")
                
                if not has_device_config:
                    errors.append(f"模型 '{key}' 没有有效的设备配置")

        except Exception as e:
            errors.append(f"配置验证失败: {str(e)}")
        return errors

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
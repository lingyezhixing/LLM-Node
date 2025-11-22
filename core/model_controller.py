"""
模型控制器 - 节点版
负责模型的启动、停止和资源管理 (无数据库依赖)
"""

import time
import threading
import os
import concurrent.futures
import queue
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from utils.logger import get_logger
from .plugin_system import PluginManager
from .config_manager import ConfigManager
from .process_manager import get_process_manager

logger = get_logger(__name__)

class LogManager:
    """简易日志管理器 - 仅用于内存转发"""
    def __init__(self):
        self.model_logs = {}
        self.lock = threading.Lock()

class ModelStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    INIT_SCRIPT = "init_script"
    HEALTH_CHECK = "health_check"
    ROUTING = "routing"
    FAILED = "failed"

class ModelController:
    """节点模型控制器"""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.models_state: Dict[str, Dict[str, Any]] = {}
        self.is_running = True
        self.plugin_manager: Optional[PluginManager] = None
        self.process_manager = get_process_manager()
        self.log_manager = LogManager()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.startup_locks: Dict[str, threading.Lock] = {}
        
        # 启动空闲检查
        self.idle_check_thread = threading.Thread(target=self.idle_check_loop, daemon=True)
        self.idle_check_thread.start()

        # 初始化状态
        for primary_name in self.config_manager.get_model_names():
            self.models_state[primary_name] = {
                "process": None,
                "status": ModelStatus.STOPPED.value,
                "last_access": None,
                "pid": None,
                "lock": threading.RLock(),
                "current_config": None,
                "failure_reason": None
            }
            self.startup_locks[primary_name] = threading.Lock()

        self.load_plugins()

    def load_plugins(self):
        device_dir = self.config_manager.get_device_plugin_dir()
        interface_dir = self.config_manager.get_interface_plugin_dir()
        self.plugin_manager = PluginManager(device_dir, interface_dir)
        self.plugin_manager.load_all_plugins(model_manager=self)
        self.plugin_manager.start_monitor()

    def start_model(self, primary_name: str) -> Tuple[bool, str]:
        """启动模型"""
        state = self.models_state[primary_name]
        model_lock = self.startup_locks[primary_name]

        with state['lock']:
            if state['status'] == ModelStatus.ROUTING.value:
                state['last_access'] = time.time() # 刷新活跃时间
                return True, f"模型 '{primary_name}' 已在运行"
            elif state['status'] == ModelStatus.STARTING.value:
                return self._wait_for_model_startup(primary_name, state)

        if not model_lock.acquire(blocking=True, timeout=60):
            return False, f"获取启动锁超时: {primary_name}"

        try:
            with state['lock']:
                if state['status'] == ModelStatus.ROUTING.value:
                    return True, "模型已由其他线程启动"
                
                state['status'] = ModelStatus.STARTING.value
                state['failure_reason'] = None
            
            return self._start_model_intelligent(primary_name)
        except Exception as e:
            with state['lock']:
                state['status'] = ModelStatus.FAILED.value
                state['failure_reason'] = str(e)
            logger.error(f"启动失败: {e}", exc_info=True)
            return False, str(e)
        finally:
            model_lock.release()

    def _wait_for_model_startup(self, primary_name, state):
        # 简化版等待逻辑
        for _ in range(120):
            with state['lock']:
                if state['status'] == ModelStatus.ROUTING.value:
                    return True, "启动成功"
                if state['status'] in [ModelStatus.FAILED.value, ModelStatus.STOPPED.value]:
                    return False, "启动失败或被停止"
            time.sleep(1)
        return False, "等待启动超时"

    def _start_model_intelligent(self, primary_name: str) -> Tuple[bool, str]:
        state = self.models_state[primary_name]
        
        # 1. 获取自适应配置
        online_devices = self.plugin_manager.get_cached_online_devices()
        
        # 如果禁用监控，伪造在线设备
        if self.config_manager.is_gpu_monitoring_disabled():
            base_config = self.config_manager.get_model_config(primary_name)
            online_devices = set()
            if base_config:
                for val in base_config.values():
                    if isinstance(val, dict) and "required_devices" in val:
                        online_devices.update(val["required_devices"])

        model_config = self.config_manager.get_adaptive_model_config(primary_name, online_devices)
        if not model_config:
            return False, "没有适合当前设备的配置方案"

        state['current_config'] = model_config

        # 2. 资源检查与释放
        if not self._check_and_free_resources(model_config):
            return False, "设备资源不足且无法释放"

        # 3. 启动进程
        with state['lock']:
            state['status'] = ModelStatus.INIT_SCRIPT.value
        
        logger.info(f"正在启动: {primary_name} (方案: {model_config.get('config_source')})")
        
        project_root = os.path.dirname(os.path.abspath(self.config_manager.config_path))
        
        def output_callback(stream, msg):
            self.log_manager.add_console_log(primary_name, msg)

        success, msg, pid = self.process_manager.start_process(
            name=f"model_{primary_name}",
            command=model_config['bat_path'],
            cwd=project_root,
            shell=True,
            output_callback=output_callback
        )

        if not success:
            return False, msg

        state['pid'] = pid

        # 4. 健康检查
        with state['lock']:
            state['status'] = ModelStatus.HEALTH_CHECK.value
        
        return self._perform_health_checks(primary_name, model_config)

    def _check_and_free_resources(self, model_config):
        if self.config_manager.is_gpu_monitoring_disabled():
            return True

        required_memory = model_config.get("memory_mb", {})
        
        for attempt in range(2):
            device_status_map = self.plugin_manager.get_device_status_snapshot()
            resource_ok = True
            deficit_devices = {}

            for dev_name, req_mb in required_memory.items():
                status = device_status_map.get(dev_name)
                if not status or not status.get('online'):
                    resource_ok = False; break
                
                info = status.get('info')
                if info and info.get('available_memory_mb', 0) < req_mb:
                    deficit_devices[dev_name] = req_mb - info.get('available_memory_mb', 0)
                    resource_ok = False
            
            if resource_ok: return True

            # 尝试释放资源
            if attempt == 0:
                if not self._stop_idle_models_for_resources(deficit_devices):
                    break
                time.sleep(3) # 等待释放
        
        return False

    def _stop_idle_models_for_resources(self, deficit_devices):
        # 查找占用相关设备的空闲模型并停止
        candidates = []
        for name, state in self.models_state.items():
            with state['lock']:
                if state['status'] == ModelStatus.ROUTING.value:
                    cfg = state.get('current_config', {})
                    used = set(cfg.get('required_devices', []))
                    if not used.isdisjoint(deficit_devices.keys()):
                        candidates.append(name)
        
        # 按最后访问时间排序
        candidates.sort(key=lambda m: self.models_state[m].get('last_access', 0) or 0)
        
        for name in candidates:
            logger.info(f"为释放资源停止空闲模型: {name}")
            self.stop_model(name)
            return True # 每次只停一个，然后重试检查
            
        return False

    def _perform_health_checks(self, name, config):
        interface = self.plugin_manager.get_interface_plugin(config.get("mode", "Chat"))
        if interface:
            success, msg = interface.health_check(name, config['port'])
            if success:
                state = self.models_state[name]
                with state['lock']:
                    state['status'] = ModelStatus.ROUTING.value
                    state['last_access'] = time.time()
                return True, "Started"
            else:
                self.stop_model(name)
                return False, msg
        return False, "No interface plugin"

    def stop_model(self, primary_name: str) -> Tuple[bool, str]:
        """停止模型"""
        state = self.models_state[primary_name]
        with state['lock']:
            if state['status'] == ModelStatus.STOPPED.value:
                return True, "Already stopped"
            
            state['status'] = ModelStatus.STOPPED.value
            state['failure_reason'] = "User requested"
            
            pid = state.get('pid')
            if pid:
                self.process_manager.stop_process(f"model_{primary_name}", force=True)
            
            state['pid'] = None
            state['current_config'] = None
            
        return True, "Stopped"

    def unload_all_models(self):
        logger.info("正在卸载所有模型...")
        for name in self.models_state:
            self.stop_model(name)

    def idle_check_loop(self):
        while self.is_running:
            time.sleep(30)
            alive_time = self.config_manager.get_alive_time() * 60
            if alive_time <= 0: continue
            
            now = time.time()
            for name, state in self.models_state.items():
                # 在节点模式下，简单的检查即可
                if state['status'] == ModelStatus.ROUTING.value and state['last_access']:
                    if (now - state['last_access']) > alive_time:
                        logger.info(f"模型 {name} 空闲超时，正在关闭...")
                        self.stop_model(name)

    def get_model_list(self):
        data = []
        for name in self.models_state:
            cfg = self.config_manager.get_model_config(name)
            if cfg:
                data.append({
                    "id": name,
                    "object": "model",
                    "mode": cfg.get("mode")
                })
        return {"object": "list", "data": data}

    def shutdown(self):
        self.is_running = False
        if self.plugin_manager:
            self.plugin_manager.stop_monitor()
        self.unload_all_models()
        self.executor.shutdown(wait=True)
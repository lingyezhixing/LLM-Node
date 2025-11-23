"""
模型控制器 - 节点版 (带文件日志管理 + 实时流支持)
"""

import time
import threading
import os
import glob
import concurrent.futures
from datetime import datetime
from typing import Dict, Tuple, Any, List, Callable
from enum import Enum
from utils.logger import get_logger
from .plugin_system import PluginManager
from .config_manager import ConfigManager
from .process_manager import get_process_manager

logger = get_logger(__name__)

class LogManager:
    """
    日志管理器：支持文件持久化 + 实时内存广播
    """
    def __init__(self, base_log_dir: str = "logs/model_logs"):
        self.base_log_dir = base_log_dir
        self.active_log_paths: Dict[str, str] = {}
        # 订阅者字典: {model_name: [callback_function, ...]}
        self.subscribers: Dict[str, List[Callable[[str], None]]] = {}
        self.lock = threading.Lock()

        if not os.path.exists(self.base_log_dir):
            try:
                os.makedirs(self.base_log_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"创建日志目录失败: {e}")

    def prepare_model_log(self, model_name: str):
        with self.lock:
            # 跨平台安全名称替换
            safe_name = model_name.replace(":", "_").replace("\\", "_").replace("/", "_").replace(os.sep, "_")
            model_dir = os.path.join(self.base_log_dir, safe_name)
            
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)

            log_files = glob.glob(os.path.join(model_dir, "*.log"))
            try:
                log_files.sort(key=os.path.getmtime)
            except Exception:
                pass

            while len(log_files) >= 10:
                oldest_file = log_files.pop(0)
                try:
                    os.remove(oldest_file)
                except Exception:
                    pass

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{safe_name}_{timestamp}.log"
            log_path = os.path.join(model_dir, log_filename)
            
            self.active_log_paths[model_name] = log_path
            
            try:
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== Log Start: {model_name} at {timestamp} ===\n")
            except Exception as e:
                logger.error(f"创建日志文件失败: {e}")

            return log_path

    def subscribe(self, model_name: str, callback: Callable[[str], None]):
        """
        订阅模型的实时日志
        callback: 一个接受字符串参数的函数
        """
        with self.lock:
            if model_name not in self.subscribers:
                self.subscribers[model_name] = []
            self.subscribers[model_name].append(callback)

    def unsubscribe(self, model_name: str, callback: Callable[[str], None]):
        """取消订阅"""
        with self.lock:
            if model_name in self.subscribers:
                try:
                    self.subscribers[model_name].remove(callback)
                    if not self.subscribers[model_name]:
                        del self.subscribers[model_name]
                except ValueError:
                    pass

    def add_console_log(self, model_name: str, message: str):
        """
        记录日志：同时写入文件和推送给订阅者
        注意：此方法通常由 ProcessManager 的监控线程调用
        """
        time_str = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{time_str}] {message}\n"

        # 1. 写入文件
        log_path = self.active_log_paths.get(model_name)
        if log_path:
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(formatted_msg)
            except Exception:
                pass

        # 2. 广播给实时流订阅者
        # 不需要加锁，因为列表本身是引用，且 copy 操作或直接迭代通常是线程安全的，
        # 为了极度严谨，这里简单加个拷贝防止迭代时修改
        subscribers_copy = []
        with self.lock:
            if model_name in self.subscribers:
                subscribers_copy = self.subscribers[model_name][:]
        
        for callback in subscribers_copy:
            try:
                callback(formatted_msg)
            except Exception as e:
                logger.error(f"日志回调执行失败: {e}")

    def shutdown(self):
        self.active_log_paths.clear()
        with self.lock:
            self.subscribers.clear()


class ModelStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    INIT_SCRIPT = "init_script"
    HEALTH_CHECK = "health_check"
    ROUTING = "routing"
    FAILED = "failed"


class ModelController:
    # ... (ModelController 类的其余部分保持完全不变) ...
    # 只要确保 ModelController 初始化时使用了上面的新 LogManager 即可
    # 以下代码仅为上下文示意，不需要修改 __init__ 逻辑，只需确保上面的 LogManager 替换了原来的
    
    """节点模型控制器"""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.models_state: Dict[str, Dict[str, Any]] = {}
        self.is_running = True
        self.plugin_manager = None
        self.process_manager = get_process_manager()
        self.log_manager = LogManager()  # 使用新的 LogManager
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.startup_locks: Dict[str, threading.Lock] = {}
        
        self.idle_check_thread = threading.Thread(target=self.idle_check_loop, daemon=True)
        self.idle_check_thread.start()

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
    
    # ... (ModelController 的其余方法 start_model 等保持不变，直接复制之前的代码即可) ...
    def load_plugins(self):
        device_dir = self.config_manager.get_device_plugin_dir()
        interface_dir = self.config_manager.get_interface_plugin_dir()
        self.plugin_manager = PluginManager(device_dir, interface_dir)
        self.plugin_manager.load_all_plugins(model_manager=self)
        self.plugin_manager.start_monitor()

    def start_model(self, primary_name: str) -> Tuple[bool, str]:
        state = self.models_state[primary_name]
        model_lock = self.startup_locks[primary_name]

        with state['lock']:
            if state['status'] == ModelStatus.ROUTING.value:
                state['last_access'] = time.time()
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
        
        online_devices = self.plugin_manager.get_cached_online_devices()
        
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

        # 资源检查与释放
        if not self._check_and_free_resources(model_config):
            return False, "设备资源不足且无法释放"

        self.log_manager.prepare_model_log(primary_name)

        with state['lock']:
            state['status'] = ModelStatus.INIT_SCRIPT.value
        
        logger.info(f"正在启动: {primary_name} (方案: {model_config.get('config_source')})")
        
        project_root = os.path.dirname(os.path.abspath(self.config_manager.config_path))
        
        def output_callback(stream, msg):
            prefix = "[ERR] " if stream == 'stderr' else ""
            self.log_manager.add_console_log(primary_name, f"{prefix}{msg}")

        success, msg, pid = self.process_manager.start_process(
            name=f"model_{primary_name}",
            command=model_config['script_path'], 
            cwd=project_root,
            shell=True, 
            capture_output=True, 
            output_callback=output_callback
        )

        if not success:
            return False, msg

        state['pid'] = pid

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
                available = info.get('available_memory_mb', 0) if info else 0
                if available < req_mb:
                    deficit_devices[dev_name] = req_mb - available
                    resource_ok = False
            
            if resource_ok: return True

            if attempt == 0:
                if not self._stop_idle_models_for_resources(deficit_devices):
                    break
                
                logger.info("等待3秒让系统回收资源...")
                time.sleep(3)
                
                if hasattr(self.plugin_manager, '_update_device_status_once'):
                    try:
                        logger.info("正在强制刷新硬件状态缓存...")
                        self.plugin_manager._update_device_status_once()
                    except Exception as e:
                        logger.warning(f"强制刷新设备状态失败: {e}")
        
        return False

    def _stop_idle_models_for_resources(self, deficit_devices):
        candidates = []
        for name, state in self.models_state.items():
            with state['lock']:
                if state['status'] == ModelStatus.ROUTING.value:
                    cfg = state.get('current_config', {})
                    used = set(cfg.get('required_devices', []))
                    if not used.isdisjoint(deficit_devices.keys()):
                        candidates.append(name)
        
        candidates.sort(key=lambda m: self.models_state[m].get('last_access', 0) or 0)
        
        for name in candidates:
            logger.info(f"为释放资源停止空闲模型: {name}")
            self.stop_model(name)
            return True 
            
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
        self.log_manager.shutdown()
        self.executor.shutdown(wait=True)
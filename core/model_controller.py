"""
模型控制器 - 节点版 (带文件日志管理 + 实时流支持 + 动态加权淘汰 + Checkpoint中断)
"""

import time
import threading
import os
import glob
import subprocess
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
    """节点模型控制器"""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.models_state: Dict[str, Dict[str, Any]] = {}
        self.is_running = True
        self.plugin_manager = None
        self.process_manager = get_process_manager()
        self.log_manager = LogManager()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.startup_locks: Dict[str, threading.Lock] = {}
        self.api_router = None  # 添加 API Router 引用

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

    def set_api_router(self, api_router):
        """注入 API Router 以获取请求状态"""
        self.api_router = api_router

    def load_plugins(self):
        """加载设备插件和接口插件"""
        device_dir = self.config_manager.get_device_plugin_dir()
        interface_dir = self.config_manager.get_interface_plugin_dir()

        # 创建插件管理器
        self.plugin_manager = PluginManager(device_dir, interface_dir)

        # 加载所有插件
        try:
            result = self.plugin_manager.load_all_plugins(model_manager=self)
            logger.info(f"设备插件自动加载完成: {list(self.plugin_manager.get_all_device_plugins().keys())}")
            logger.info(f"接口插件自动加载完成: {list(self.plugin_manager.get_all_interface_plugins().keys())}")

            logger.info("正在初始化设备状态缓存...")
            self.plugin_manager.update_device_status()

            # 检查是否有设备在线 (使用缓存读取)
            online_devices = self.plugin_manager.get_cached_online_devices()
            if online_devices:
                logger.info(f"在线设备: {list(online_devices)}")
            else:
                logger.warning("未检测到在线设备")

        except Exception as e:
            logger.error(f"自动加载插件失败: {e}")
            raise

    def _check_if_cancelled(self, primary_name: str) -> bool:
        """[Checkpoint] 检查启动流程是否已被用户取消"""
        state = self.models_state[primary_name]
        with state['lock']:
            cancelled = state['status'] == ModelStatus.STOPPED.value
            if cancelled:
                logger.info(f"检测到取消信号，终止启动流程: {primary_name}")
                state['failure_reason'] = "启动被用户中断"
        return cancelled

    def _reset_model_state(self, state: Dict[str, Any]):
        """重置状态字典为初始值"""
        state.update({
            "process": None,
            "pid": None,
            "status": ModelStatus.STOPPED.value,
            "last_access": None,
            "current_config": None,
            "failure_reason": None
        })

    def start_auto_start_models(self):
        """批量启动配置为自动启动的模型"""
        logger.info("正在扫描自动启动配置...")

        if not self.config_manager.is_gpu_monitoring_disabled():
            self.plugin_manager.update_device_status()

        online_devices = self.plugin_manager.get_cached_online_devices()

        if not online_devices and not self.config_manager.is_gpu_monitoring_disabled():
            logger.warning("未检测到在线设备，跳过自动启动")
            return

        auto_start_models = [
            name for name in self.config_manager.get_model_names()
            if self.config_manager.is_auto_start(name)
        ]

        if not auto_start_models:
            logger.info("无自动启动模型")
            return

        logger.info(f"准备并行启动 {len(auto_start_models)} 个模型: {auto_start_models}")

        def start_single_model(model_name):
            try:
                success, msg = self.start_model(model_name)
                return model_name, success, msg
            except Exception as ex:
                logger.error(f"自动启动异常 [{model_name}]: {ex}")
                return model_name, False, f"异常: {ex}"

        futures = []
        for model_name in auto_start_models:
            futures.append(self.executor.submit(start_single_model, model_name))

        started_count = 0
        for future in concurrent.futures.as_completed(futures, timeout=120):
            try:
                name, success, msg = future.result()
                if success:
                    started_count += 1
                else:
                    logger.warning(f"自动启动失败 [{name}]: {msg}")
            except Exception as e:
                logger.error(f"处理启动结果异常: {e}")

        logger.info(f"自动启动流程完成: 成功 {started_count}/{len(auto_start_models)}")

    def start_model(self, primary_name: str) -> Tuple[bool, str]:
        """
        启动模型（线程安全，支持重入与中断）
        """
        state = self.models_state[primary_name]
        model_lock = self.startup_locks[primary_name]

        # 1. 快速状态检查
        with state['lock']:
            if state['status'] == ModelStatus.ROUTING.value:
                state['last_access'] = time.time()
                return True, f"模型 '{primary_name}' 已在运行"
            elif state['status'] == ModelStatus.STARTING.value:
                return self._wait_for_model_startup(primary_name, state)

        # 2. 获取启动互斥锁
        lock_acquired = False
        try:
            lock_acquired = model_lock.acquire(blocking=True, timeout=60)
            if not lock_acquired:
                return False, f"获取启动锁超时: {primary_name}"
        except Exception as e:
            if lock_acquired:
                model_lock.release()
            return False, f"锁获取异常: {e}"

        try:
            with state['lock']:
                # 双重检查
                if state['status'] == ModelStatus.ROUTING.value:
                    return True, "模型已由其他线程启动"

                state['status'] = ModelStatus.STARTING.value
                state['failure_reason'] = None

            try:
                success, message = self._start_model_intelligent(primary_name)

                # [Checkpoint] 最终防线
                if self._check_if_cancelled(primary_name):
                    with state['lock']:
                        pid = state.get('pid')
                        if pid:
                            logger.warning(f"启动完成但收到停止信号，清理残留进程 PID: {pid}")
                            self.process_manager.stop_process(f"model_{primary_name}", force=True)
                            self._reset_model_state(state)
                    return False, "启动完成后被立即停止"

                if success:
                    logger.info(f"模型 '{primary_name}' 启动成功，正在刷新设备状态缓存...")
                    self.plugin_manager.update_device_status()

                return success, message
            except Exception as e:
                with state['lock']:
                    state['status'] = ModelStatus.FAILED.value
                    state['failure_reason'] = str(e)
                logger.error(f"启动失败: {e}", exc_info=True)
                return False, f"启动异常: {e}"

        finally:
            if lock_acquired:
                try:
                    model_lock.release()
                except Exception:
                    pass

    def _wait_for_model_startup(self, primary_name, state):
        wait_start = time.time()
        max_wait = 120
        last_log = wait_start

        while True:
            now = time.time()
            elapsed = now - wait_start

            if now - last_log >= 30:
                logger.info(f"等待中... {primary_name} ({elapsed:.1f}s)")
                last_log = now

            with state['lock']:
                status = state['status']
                fail_reason = state.get('failure_reason')

            if elapsed > max_wait:
                return False, "等待启动超时"

            if status == ModelStatus.ROUTING.value:
                return True, "启动成功"
            elif status == ModelStatus.FAILED.value:
                return False, f"启动失败: {fail_reason}"
            elif status == ModelStatus.STOPPED.value:
                logger.info(f"等待期间检测到停止信号: {primary_name}")
                return False, "启动被用户中断"

            time.sleep(0.5)

    def _start_model_intelligent(self, primary_name: str) -> Tuple[bool, str]:
        state = self.models_state[primary_name]

        try:
            # 刷新设备状态缓存
            if not self.config_manager.is_gpu_monitoring_disabled():
                self.plugin_manager.update_device_status()

            online_devices = self.plugin_manager.get_cached_online_devices()

            if self.config_manager.is_gpu_monitoring_disabled():
                base_config = self.config_manager.get_model_config(primary_name)
                online_devices = set()
                if base_config:
                    for val in base_config.values():
                        if isinstance(val, dict) and "required_devices" in val:
                            online_devices.update(val["required_devices"])

            # [Checkpoint 2] 设备检查后
            if self._check_if_cancelled(primary_name):
                return False, "启动中断（阶段2）"

            model_config = self.config_manager.get_adaptive_model_config(primary_name, online_devices)
            if not model_config:
                error_msg = "没有适合当前设备的配置方案"
                with state['lock']:
                    state['status'] = ModelStatus.FAILED.value
                    state['failure_reason'] = error_msg
                return False, error_msg

            state['current_config'] = model_config

            # 资源检查与释放
            if not self._check_and_free_resources(model_config):
                error_msg = "设备资源不足且无法释放"
                with state['lock']:
                    state['status'] = ModelStatus.FAILED.value
                    state['failure_reason'] = error_msg
                return False, error_msg

            # [Checkpoint 3] 资源检查后
            if self._check_if_cancelled(primary_name):
                return False, "启动中断（阶段3）"

            self.log_manager.prepare_model_log(primary_name)

            with state['lock']:
                state['status'] = ModelStatus.INIT_SCRIPT.value

            logger.info(f"正在启动: {primary_name} (方案: {model_config.get('config_source')})")

            project_root = os.path.dirname(os.path.abspath(self.config_manager.config_path))

            def output_callback(stream, msg):
                self.log_manager.add_console_log(primary_name, msg)

            # Windows 进程组标志
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else None

            success, msg, pid = self.process_manager.start_process(
                name=f"model_{primary_name}",
                command=model_config['script_path'],
                cwd=project_root,
                description=f"模型进程: {primary_name}",
                shell=True,
                creation_flags=creation_flags,
                capture_output=True,
                output_callback=output_callback
            )

            if not success:
                with state['lock']:
                    state['status'] = ModelStatus.FAILED.value
                    state['failure_reason'] = msg
                return False, f"进程启动失败: {msg}"

            # [Checkpoint 4] 进程启动后立即检查
            with state['lock']:
                if state['status'] == ModelStatus.STOPPED.value:
                    logger.warning(f"进程已启动(PID {pid})但收到停止请求，立即终止")
                    self.process_manager.stop_process(f"model_{primary_name}", force=True)
                    return False, "启动中断（进程已终止）"

                state['pid'] = pid

            # 健康检查
            return self._perform_health_checks(primary_name, model_config)

        except Exception as e:
            logger.error(f"智能启动流程异常: {e}", exc_info=True)
            with state['lock']:
                state['status'] = ModelStatus.FAILED.value
                state['failure_reason'] = str(e)
            return False, f"启动异常: {e}"

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

                try:
                    logger.info("正在强制刷新硬件状态缓存...")
                    self.plugin_manager.update_device_status()
                except Exception as e:
                    logger.warning(f"强制刷新设备状态失败: {e}")

        return False

    def _stop_idle_models_for_resources(self, deficit_devices) -> bool:
        """
        [优化版] 停止一个空闲模型以释放资源
        实现动态加权淘汰算法：分数 = 空闲时间 / max(0.5, 显存占用GB)
        优先关闭：空闲时间长 且 显存占用小（重启成本低）的模型
        """
        idle_candidates = []
        now = time.time()

        for name, state in self.models_state.items():
            with state['lock']:
                # 1. 状态检查
                if state['status'] != ModelStatus.ROUTING.value:
                    continue

                # 2. 活跃请求检查 (防止误杀正在工作的模型)
                if self.api_router and self.api_router.pending_requests.get(name, 0) > 0:
                    logger.debug(f"跳过模型 {name}: 有待处理请求")
                    continue

                current_config = state.get('current_config')
                if not current_config:
                    continue

                # 3. 设备相关性检查
                used_devices = set(current_config.get('required_devices', []))
                if not used_devices:
                    used_devices = set(current_config.get('memory_mb', {}).keys())

                if used_devices.isdisjoint(set(deficit_devices.keys())):
                    continue

                # 4. 计算淘汰评分
                last_access = state.get('last_access') or 0
                idle_seconds = max(0, now - last_access)

                total_memory_mb = sum(current_config.get('memory_mb', {}).values())
                memory_gb = total_memory_mb / 1024.0
                # 设定0.5GB作为分母下限，防止除以极小值导致分数过大
                memory_gb_for_score = max(0.5, memory_gb)

                # 核心公式：显存越小、空闲越久，分数越高 -> 越容易被关闭
                # 理念：保留大显存模型（重启慢），优先清理小模型
                eviction_score = idle_seconds / memory_gb_for_score

                idle_candidates.append({
                    "name": name,
                    "score": eviction_score,
                    "idle_seconds": idle_seconds,
                    "memory_gb": memory_gb
                })

        if not idle_candidates:
            logger.info("没有找到占用相关设备的可停止空闲模型")
            return False

        # 按分数从高到低排序
        sorted_candidates = sorted(idle_candidates, key=lambda x: x['score'], reverse=True)

        logger.info(f"资源释放候选列表 (按优先级排序):")
        for c in sorted_candidates:
            logger.info(f"  - {c['name']}: 空闲 {c['idle_seconds']:.0f}s, 显存 {c['memory_gb']:.2f}GB -> 分数 {c['score']:.4f}")

        # 关闭分数最高的模型
        candidate_to_stop = sorted_candidates[0]
        model_name = candidate_to_stop['name']
        logger.info(f"为释放资源，正在停止模型: {model_name} (当前评分最高)")
        success, message = self.stop_model(model_name)

        if success:
            logger.info(f"模型 {model_name} 已成功停止")
            return True
        else:
            logger.warning(f"尝试停止模型 {model_name} 失败: {message}")
            return False

    def _perform_health_checks(self, name, config):
        # [Checkpoint 5] 健康检查前
        if self._check_if_cancelled(name):
            return False, "启动中断（健康检查前）"

        state = self.models_state[name]
        with state['lock']:
            state['status'] = ModelStatus.HEALTH_CHECK.value

        interface = self.plugin_manager.get_interface_plugin(config.get("mode", "Chat"))
        if interface:
            success, msg = interface.health_check(name, config['port'])

            # [Checkpoint 6] 健康检查后
            if self._check_if_cancelled(name):
                return False, "启动中断（健康检查后）"

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

    def stop_model(self, primary_name: str, refresh_cache: bool = True) -> Tuple[bool, str]:
        """停止模型（支持中断启动中的模型）"""
        state = self.models_state[primary_name]

        logger.info(f"收到停止请求: {primary_name}")

        # 1. 设置信号
        with state['lock']:
            current_status = state['status']
            if current_status in [ModelStatus.STOPPED.value, ModelStatus.FAILED.value]:
                return True, "模型已停止"

            state['status'] = ModelStatus.STOPPED.value
            state['failure_reason'] = "被用户请求停止"
            pid = state.get('pid')

        # 2. 终止进程
        if pid:
            self.process_manager.stop_process(f"model_{primary_name}", force=True)

        # 3. 清理状态
        with state['lock']:
            self._reset_model_state(state)

        # 4. 刷新设备缓存
        if refresh_cache:
            logger.info(f"模型 '{primary_name}' 已停止，正在刷新设备状态缓存...")
            self.plugin_manager.update_device_status()

        return True, "Stopped"

    def unload_all_models(self):
        """并行卸载所有运行中的模型"""
        logger.info("开始卸载所有模型...")
        models_to_stop = []

        # 筛选运行中的模型（跳过已停止/失败的）
        for name, state in self.models_state.items():
            with state['lock']:
                if state['status'] not in [ModelStatus.STOPPED.value, ModelStatus.FAILED.value]:
                    models_to_stop.append(name)

        if not models_to_stop:
            logger.info("无运行中的模型")
            return 0

        def stop_task(name):
            try:
                ok, msg = self.stop_model(name, refresh_cache=False)
                return name, ok, msg
            except Exception as e:
                return name, False, str(e)

        futures = [self.executor.submit(stop_task, name) for name in models_to_stop]
        stopped_count = 0

        timeout = len(models_to_stop) * 5 + 10
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                name, ok, msg = future.result()
                if ok:
                    stopped_count += 1
                else:
                    logger.warning(f"卸载失败 [{name}]: {msg}")
            except Exception as e:
                logger.error(f"处理结果异常: {e}")

        # 所有模型停止后，只刷新一次缓存
        if stopped_count > 0:
            self.plugin_manager.update_device_status()

        logger.info(f"卸载完成: {stopped_count}/{len(models_to_stop)}")
        return stopped_count

    def idle_check_loop(self):
        """
        空闲检查循环 - 包含双重检查机制防止竞态条件
        """
        while self.is_running:
            time.sleep(30)
            # 获取配置的存活时间（分钟 -> 秒）
            alive_time_min = self.config_manager.get_alive_time()
            if alive_time_min <= 0:
                continue

            alive_time = alive_time_min * 60
            now = time.time()
            models_to_stop = []

            # 第一阶段：筛选候选模型
            for name in list(self.models_state.keys()):
                state = self.models_state[name]
                with state['lock']:
                    if state['status'] != ModelStatus.ROUTING.value:
                        continue

                    last_access = state.get('last_access')
                    if not last_access:
                        continue

                    # 检查是否有待处理请求
                    pending_count = 0
                    if self.api_router:
                        pending_count = self.api_router.pending_requests.get(name, 0)

                    # 只有无请求且超时才标记
                    if pending_count == 0 and (now - last_access) > alive_time:
                        models_to_stop.append(name)

            # 第二阶段：执行关闭（带最终确认）
            for name in models_to_stop:
                # 【关键优化】在真正下刀之前，再次确认请求数
                # 防止在筛选和执行的间隙有新请求进来
                should_stop = True
                if self.api_router:
                    current_pending = self.api_router.pending_requests.get(name, 0)
                    if current_pending > 0:
                        logger.info(f"模型 {name} 在关闭前一刻收到新请求，取消关闭")
                        should_stop = False

                if should_stop:
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

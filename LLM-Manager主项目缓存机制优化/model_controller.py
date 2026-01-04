"""
模型控制器 - 负责模型的启动、停止和资源管理
优化版本：支持并行启动和智能资源管理
修复：解决自动关闭时的死锁问题 (Lock -> RLock, 优化空闲检查逻辑)
适配：Linux 路径与执行逻辑兼容
"""

import subprocess
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
from .data_manager import Monitor

logger = get_logger(__name__)


class LogManager:
    """日志管理器 - 负责模型控制台日志的收集、存储和流式推送"""

    def __init__(self):
        """初始化日志管理器"""
        self.model_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.log_subscribers: Dict[str, List[queue.Queue]] = {}
        self.log_locks: Dict[str, threading.Lock] = {}
        self.global_lock = threading.Lock()
        self._running = True

        logger.info("日志管理器初始化完成")

    def register_model(self, model_name: str):
        """注册模型日志管理"""
        with self.global_lock:
            if model_name not in self.model_logs:
                self.model_logs[model_name] = []
                self.log_subscribers[model_name] = []
                self.log_locks[model_name] = threading.Lock()
                logger.debug(f"已注册模型日志管理: {model_name}")

    def unregister_model(self, model_name: str):
        """注销模型日志管理"""
        with self.global_lock:
            if model_name in self.model_logs:
                # 通知所有订阅者连接关闭
                for subscriber_queue in self.log_subscribers[model_name]:
                    try:
                        subscriber_queue.put(None)  # 发送结束信号
                    except:
                        pass

                del self.model_logs[model_name]
                del self.log_subscribers[model_name]
                del self.log_locks[model_name]
                logger.debug(f"已注销模型日志管理: {model_name}")

    def add_console_log(self, model_name: str, message: str):
        """添加模型控制台日志"""
        if model_name not in self.model_logs:
            self.register_model(model_name)

        log_entry = {
            "timestamp": time.time(),
            "message": message
        }

        model_lock = self.log_locks[model_name]
        with model_lock:
            self.model_logs[model_name].append(log_entry)

        # 通知订阅者
        self._notify_subscribers(model_name, log_entry)

    def get_logs(self, model_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取模型日志"""
        if model_name not in self.model_logs:
            return []

        model_lock = self.log_locks[model_name]
        with model_lock:
            logs = self.model_logs[model_name].copy()

        if limit is not None:
            logs = logs[-limit:]

        return logs

    def get_all_logs(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取所有模型日志"""
        with self.global_lock:
            result = {}
            for model_name in self.model_logs:
                result[model_name] = self.get_logs(model_name)
            return result

    def clear_logs(self, model_name: str):
        """清空模型日志"""
        if model_name in self.model_logs:
            model_lock = self.log_locks[model_name]
            with model_lock:
                self.model_logs[model_name].clear()
            logger.debug(f"已清空模型日志: {model_name}")

    def cleanup_old_logs(self, model_name: str, keep_minutes: int) -> int:
        """
        清理指定分钟数之前的日志

        Args:
            model_name: 模型名称
            keep_minutes: 保留最近多少分钟的日志

        Returns:
            删除的日志条目数量
        """
        if model_name not in self.model_logs:
            return 0

        current_time = time.time()
        cutoff_time = current_time - (keep_minutes * 60)

        model_lock = self.log_locks[model_name]
        with model_lock:
            original_count = len(self.model_logs[model_name])

            # 保留在截止时间之后的日志
            self.model_logs[model_name] = [
                log for log in self.model_logs[model_name]
                if log['timestamp'] > cutoff_time
            ]

            removed_count = original_count - len(self.model_logs[model_name])

        if removed_count > 0:
            logger.debug(f"已清理模型 '{model_name}' {keep_minutes} 分钟前的日志，删除 {removed_count} 条")

        return removed_count

    def subscribe_to_logs(self, model_name: str) -> queue.Queue:
        """订阅模型日志流"""
        if model_name not in self.model_logs:
            self.register_model(model_name)

        subscriber_queue = queue.Queue()

        with self.global_lock:
            self.log_subscribers[model_name].append(subscriber_queue)

        logger.debug(f"新订阅者加入模型日志流: {model_name}")
        return subscriber_queue

    def unsubscribe_from_logs(self, model_name: str, subscriber_queue: queue.Queue):
        """取消订阅模型日志流"""
        if model_name in self.log_subscribers:
            with self.global_lock:
                if subscriber_queue in self.log_subscribers[model_name]:
                    self.log_subscribers[model_name].remove(subscriber_queue)
                    logger.debug(f"订阅者离开模型日志流: {model_name}")

    def _notify_subscribers(self, model_name: str, log_entry: Dict[str, Any]):
        """通知所有订阅者新的日志条目"""
        if model_name not in self.log_subscribers:
            return

        # 创建副本以避免在迭代时修改列表
        subscribers_copy = self.log_subscribers[model_name].copy()

        for subscriber_queue in subscribers_copy:
            try:
                # 非阻塞方式发送，如果队列满则跳过
                subscriber_queue.put_nowait(log_entry)
            except queue.Full:
                # 订阅者队列已满，移除该订阅者
                with self.global_lock:
                    if subscriber_queue in self.log_subscribers[model_name]:
                        self.log_subscribers[model_name].remove(subscriber_queue)
                        logger.debug(f"移除已满的订阅者队列: {model_name}")
            except Exception:
                # 其他错误，移除订阅者
                with self.global_lock:
                    if subscriber_queue in self.log_subscribers[model_name]:
                        self.log_subscribers[model_name].remove(subscriber_queue)
                        logger.debug(f"移除异常的订阅者队列: {model_name}")

    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        with self.global_lock:
            stats = {
                "total_models": len(self.model_logs),
                "total_log_entries": sum(len(logs) for logs in self.model_logs.values()),
                "total_subscribers": sum(len(subscribers) for subscribers in self.log_subscribers.values()),
                "model_stats": {}
            }

            for model_name, logs in self.model_logs.items():
                stats["model_stats"][model_name] = {
                    "log_count": len(logs),
                    "subscriber_count": len(self.log_subscribers[model_name])
                }

            return stats

    def shutdown(self):
        """关闭日志管理器"""
        self._running = False

        # 通知所有订阅者
        with self.global_lock:
            for model_name, subscribers in self.log_subscribers.items():
                for subscriber_queue in subscribers:
                    try:
                        subscriber_queue.put(None)  # 发送结束信号
                    except:
                        pass

        logger.info("日志管理器已关闭")


class ModelRuntimeMonitor:
    """模型运行时间监控器 - 负责记录模型运行时间到数据库"""

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化模型运行时间监控器

        Args:
            db_path: 数据库路径，默认使用monitor的默认路径
        """
        self.monitor = Monitor(db_path)
        self.active_models: Dict[str, threading.Timer] = {}
        self.lock = threading.Lock()

        logger.info("模型运行时间监控器初始化完成")

    def record_model_start(self, model_name: str) -> bool:
        """
        记录模型启动时间并启动定时器

        Args:
            model_name: 模型名称

        Returns:
            是否成功记录
        """
        start_time = time.time()

        # 记录启动时间到数据库
        try:
            self.monitor.add_model_runtime_start(model_name, start_time)
            logger.info(f"已记录模型 '{model_name}' 启动时间: {start_time}")
        except Exception as e:
            logger.error(f"记录模型 '{model_name}' 启动时间失败: {e}")
            return False

        # 启动定时更新线程
        with self.lock:
            # 停止已有的定时器（如果有）
            if model_name in self.active_models:
                self.active_models[model_name].cancel()

            # 创建新的定时器，每10秒更新一次
            timer = threading.Timer(10.0, self._update_runtime_periodically, args=[model_name])
            timer.daemon = True
            timer.start()
            self.active_models[model_name] = timer

        logger.info(f"已启动模型 '{model_name}' 的运行时间定时更新器")
        return True

    def _update_runtime_periodically(self, model_name: str):
        """
        定期更新模型运行时间

        Args:
            model_name: 模型名称
        """
        with self.lock:
            # 如果模型仍在活动状态，继续定时更新
            if model_name in self.active_models:
                end_time = time.time()
                try:
                    self.monitor.update_model_runtime_end(model_name, end_time)
                    logger.debug(f"已更新模型 '{model_name}' 运行时间: {end_time}")
                except Exception as e:
                    logger.error(f"更新模型 '{model_name}' 运行时间失败: {e}")

                # 创建下一个定时器
                timer = threading.Timer(10.0, self._update_runtime_periodically, args=[model_name])
                timer.daemon = True
                timer.start()
                self.active_models[model_name] = timer

    def record_model_stop(self, model_name: str) -> bool:
        """
        记录模型停止时间并停止定时器

        Args:
            model_name: 模型名称

        Returns:
            是否成功记录
        """
        end_time = time.time()

        # 停止定时器
        with self.lock:
            if model_name in self.active_models:
                self.active_models[model_name].cancel()
                del self.active_models[model_name]

        # 记录最终停止时间到数据库
        try:
            self.monitor.update_model_runtime_end(model_name, end_time)
            logger.info(f"已记录模型 '{model_name}' 停止时间: {end_time}")
            return True
        except Exception as e:
            logger.error(f"记录模型 '{model_name}' 停止时间失败: {e}")
            return False

    def is_model_monitored(self, model_name: str) -> bool:
        """
        检查模型是否正在被监控

        Args:
            model_name: 模型名称

        Returns:
            是否正在监控
        """
        with self.lock:
            return model_name in self.active_models

    def get_active_models(self) -> List[str]:
        """
        获取正在被监控的模型列表

        Returns:
            正在监控的模型名称列表
        """
        with self.lock:
            return list(self.active_models.keys())

    def shutdown(self):
        """关闭监控器，停止所有定时器"""
        with self.lock:
            for model_name, timer in self.active_models.items():
                timer.cancel()
                logger.info(f"已停止模型 '{model_name}' 的运行时间监控")
            self.active_models.clear()

        try:
            self.monitor.close()
            logger.info("模型运行时间监控器已关闭")
        except Exception as e:
            logger.error(f"关闭模型运行时间监控器失败: {e}")

    def __del__(self):
        """析构函数"""
        try:
            self.shutdown()
        except Exception:
            pass


class ModelStatus(Enum):
    """模型状态枚举"""
    STOPPED = "stopped"
    STARTING = "starting"
    INIT_SCRIPT = "init_script"
    HEALTH_CHECK = "health_check"
    ROUTING = "routing"
    FAILED = "failed"


class ModelController:
    """优化的模型控制器 - 支持并行启动和智能资源管理"""

    def __init__(self, config_manager: ConfigManager):
        """
        初始化模型控制器

        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager
        self.models_state: Dict[str, Dict[str, Any]] = {}
        self.is_running = True
        self.plugin_manager: Optional[PluginManager] = None
        self.process_manager = get_process_manager()
        self.runtime_monitor = ModelRuntimeMonitor()
        self.log_manager = LogManager()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.startup_futures: Dict[str, concurrent.futures.Future] = {}
        self.startup_locks: Dict[str, threading.Lock] = {}  # 每个模型的专用锁
        self.shutdown_event = threading.Event()
        self.api_router: Optional[Any] = None # 类型设为 Any 避免循环导入

        # 启动空闲检查线程
        self.idle_check_thread = threading.Thread(target=self.idle_check_loop, daemon=True)
        self.idle_check_thread.start()

        # 初始化模型状态
        new_states = {}
        model_names = self.config_manager.get_model_names()

        for primary_name in model_names:
            if primary_name in self.models_state:
                new_states[primary_name] = self.models_state[primary_name]
            else:
                new_states[primary_name] = {
                    "process": None,
                    "status": ModelStatus.STOPPED.value,
                    "last_access": None,
                    "pid": None,
                    # 【核心修复】使用 RLock 替代 Lock，允许同一线程（如 idle_check 调用 stop_model 时）重入锁
                    # 这直接解决了自动关闭时的死锁问题
                    "lock": threading.RLock(),
                    "log_thread": None,
                    "current_config": None,
                    "failure_reason": None
                }
                # 为每个模型创建专用的启动锁
                self.startup_locks[primary_name] = threading.Lock()
                # 注册模型到日志管理器
                self.log_manager.register_model(primary_name)

        self.models_state = new_states
        self.load_plugins()
    
    def set_api_router(self, api_router: Any):
        """设置API路由器实例，用于获取待处理请求数"""
        self.api_router = api_router
        logger.info("API Router 实例已成功注入到 Model Controller")

    def load_plugins(self):
        """加载设备插件和接口插件"""
        # 从配置管理器获取插件目录
        device_dir = self.config_manager.get_device_plugin_dir()
        interface_dir = self.config_manager.get_interface_plugin_dir()

        # 创建插件管理器
        self.plugin_manager = PluginManager(device_dir, interface_dir)

        # 加载所有插件
        try:
            result = self.plugin_manager.load_all_plugins(model_manager=self)
            logger.info(f"设备插件自动加载完成: {list(self.plugin_manager.get_all_device_plugins().keys())}")
            logger.info(f"接口插件自动加载完成: {list(self.plugin_manager.get_all_interface_plugins().keys())}")
            
            # 按需监控：不在启动时自动开启监控，等待首次API请求时再启动
            logger.info("设备监控设置为按需模式，将在首次请求时启动")
            # self.plugin_manager.start_monitor()  # 已禁用自动启动

            # 检查是否有设备在线 (使用缓存读取)
            online_devices = self.plugin_manager.get_cached_online_devices()
            if online_devices:
                logger.info(f"在线设备: {list(online_devices)}")
            else:
                logger.warning("未检测到在线设备")

        except Exception as e:
            logger.error(f"自动加载插件失败: {e}")

    def start_auto_start_models(self):
        """优化的启动所有标记为自动启动的模型 - 支持并行启动"""
        logger.info("检查需要自动启动的模型...")

        # 检查设备在线状态 (使用缓存)
        online_devices = self.plugin_manager.get_cached_online_devices()
        
        # 如果禁用了监控，我们跳过“无在线设备”的检查，允许尝试启动
        if not online_devices and not self.config_manager.is_gpu_monitoring_disabled():
            logger.warning("没有在线设备，跳过自动启动模型")
            return

        auto_start_models = [
            primary_name for primary_name in self.config_manager.get_model_names()
            if self.config_manager.is_auto_start(primary_name)
        ]

        if not auto_start_models:
            logger.info("没有需要自动启动的模型")
            return

        logger.info(f"准备并行启动 {len(auto_start_models)} 个自动启动模型: {auto_start_models}")

        def start_single_model(model_name):
            try:
                success, message = self.start_model(model_name)
                return model_name, success, message
            except Exception as e:
                logger.error(f"自动启动模型 {model_name} 时发生异常: {e}")
                return model_name, False, f"启动异常: {e}"

        # 使用线程池并行启动模型
        futures = []
        for model_name in auto_start_models:
            future = self.executor.submit(start_single_model, model_name)
            futures.append(future)

        # 收集结果
        started_models = []
        for future in concurrent.futures.as_completed(futures, timeout=120):
            try:
                model_name, success, message = future.result()
                if success:
                    logger.info(f"自动启动模型 {model_name} 成功")
                    started_models.append(model_name)
                else:
                    logger.error(f"自动启动模型 {model_name} 失败: {message}")
            except Exception as e:
                logger.error(f"处理模型启动结果时发生异常: {e}")

        logger.info(f"自动启动完成，成功: {len(started_models)}/{len(auto_start_models)}")
        if started_models:
            logger.info(f"成功启动的模型: {started_models}")

    def start_model(self, primary_name: str) -> Tuple[bool, str]:
        """
        优化的启动模型 - 支持并发启动和统一等待机制，接受主名称

        Args:
            primary_name: 模型主名称

        Returns:
            (成功状态, 消息)
        """
        state = self.models_state[primary_name]
        model_lock = self.startup_locks[primary_name]

        # 快速检查，避免不必要的锁等待
        with state['lock']:
            if state['status'] == ModelStatus.ROUTING.value:
                return True, f"模型 '{primary_name}' 已在运行"
            elif state['status'] == ModelStatus.STARTING.value:
                # 模型正在启动，所有请求统一等待启动完成
                logger.info(f"模型 '{primary_name}' 正在启动中，等待完成...")
                return self._wait_for_model_startup(primary_name, state)

        # 【修复】使用带超时的阻塞锁获取，避免死锁
        logger.info(f"等待启动模型 '{primary_name}' 的启动锁...")
        lock_acquired = False
        try:
            lock_acquired = model_lock.acquire(blocking=True, timeout=60)  # 60秒超时
            if not lock_acquired:
                logger.error(f"获取模型 '{primary_name}' 启动锁超时，可能存在死锁")
                # 尝试强制获取锁作为最后手段
                logger.warning(f"尝试强制获取模型 '{primary_name}' 的启动锁...")
                lock_acquired = model_lock.acquire(blocking=False)
                if not lock_acquired:
                    return False, f"获取模型 '{primary_name}' 启动锁失败，请稍后重试"
            logger.info(f"获取到模型 '{primary_name}' 的启动锁")
        except Exception as e:
            logger.error(f"获取模型 '{primary_name}' 启动锁时发生异常: {e}")
            if lock_acquired:
                model_lock.release()
            return False, f"获取启动锁时发生异常: {e}"

        try:
            logger.info(f"开始启动模型 '{primary_name}'")

            # 双重检查：在获取锁后再次确认模型状态
            with state['lock']:
                current_status = state['status']

                if current_status == ModelStatus.ROUTING.value:
                    logger.info(f"模型 '{primary_name}' 已被其他线程启动")
                    return True, f"模型 {primary_name} 已成功启动"
                elif current_status == ModelStatus.STOPPED.value:
                    # 检查是否被外部停止请求中断
                    if state.get('failure_reason') == "被用户请求停止":
                        logger.info(f"模型 '{primary_name}' 启动过程被用户请求中断")
                        return False, f"模型 '{primary_name}' 启动被用户中断"
                    logger.info(f"模型 '{primary_name}' 状态为已停止，继续启动流程")
                elif current_status == ModelStatus.FAILED.value:
                    logger.info(f"模型 '{primary_name}' 之前启动失败，重新尝试启动")
                    # 清除失败原因
                    state['failure_reason'] = None
                elif current_status == ModelStatus.STARTING.value:
                    logger.warning(f"模型 '{primary_name}' 仍处于启动状态，这可能表示存在状态不一致")
                    # 释放模型锁并等待
                    return self._wait_for_model_startup(primary_name, state)

                # 确认由当前线程执行加载
                state['status'] = ModelStatus.STARTING.value
                state['failure_reason'] = None  # 清除之前的失败原因
                # 使用系统日志记录器记录启动操作
                logger.info(f"开始启动模型 '{primary_name}' (之前状态: {current_status})")

            try:
                success, message = self._start_model_intelligent(primary_name)

                # 启动成功后刷新设备状态缓存
                if success:
                    logger.info(f"模型 '{primary_name}' 启动成功，正在刷新设备状态缓存...")
                    self.plugin_manager.update_device_status()

                return success, message
            except Exception as e:
                with state['lock']:
                    state['status'] = ModelStatus.FAILED.value
                    state['failure_reason'] = str(e)
                logger.error(f"启动模型 {primary_name} 失败: {e}", exc_info=True)
                return False, f"启动模型 {primary_name} 失败: {e}"
        finally:
            if lock_acquired:
                try:
                    model_lock.release()
                    logger.info(f"已释放模型 '{primary_name}' 的启动锁")
                except Exception as e:
                    logger.error(f"释放模型 '{primary_name}' 启动锁时发生异常: {e}")

    def _wait_for_model_startup(self, primary_name: str, state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        【增强】等待模型启动完成 - 支持更长的等待时间和更好的状态检查

        Args:
            primary_name: 模型名称
            state: 模型状态

        Returns:
            (成功状态, 消息)
        """
        wait_start = time.time()
        max_wait_time = 120  # 【增强】增加等待时间到120秒，适应大模型启动时间
        check_interval = 0.5  # 检查间隔（秒）
        last_log_time = wait_start

        logger.info(f"开始等待模型 '{primary_name}' 启动完成，最大等待时间: {max_wait_time}秒")

        while True:
            current_time = time.time()
            elapsed_time = current_time - wait_start

            # 定期输出等待日志（每30秒一次）
            if current_time - last_log_time >= 30:
                logger.info(f"持续等待模型 '{primary_name}' 启动中... 已等待 {elapsed_time:.1f}秒")
                last_log_time = current_time

            # 【增强】安全的状态检查，使用锁避免竞态条件
            with state['lock']:
                current_status = state['status']
                failure_reason = state.get('failure_reason')

            # 检查是否超时
            if elapsed_time > max_wait_time:
                logger.error(f"等待模型 '{primary_name}' 启动超时，等待时间: {elapsed_time:.1f}秒")
                return False, f"等待模型 '{primary_name}' 启动超时（超过{max_wait_time}秒）"

            # 检查启动状态
            if current_status == ModelStatus.ROUTING.value:
                logger.info(f"模型 '{primary_name}' 启动完成，总等待时间: {elapsed_time:.1f}秒")
                return True, f"模型 {primary_name} 已成功启动"
            elif current_status == ModelStatus.FAILED.value:
                error_msg = f"模型 '{primary_name}' 启动失败"
                if failure_reason:
                    error_msg += f": {failure_reason}"
                logger.error(error_msg)
                return False, error_msg
            elif current_status == ModelStatus.STOPPED.value:
                # 检查是否被用户主动停止
                if failure_reason and "被用户请求停止" in failure_reason:
                    logger.info(f"模型 '{primary_name}' 启动过程被用户中断，等待时间: {elapsed_time:.1f}秒")
                    return False, f"模型 '{primary_name}' 启动被用户中断"
                else:
                    # 意外状态，可能被外部停止
                    logger.warning(f"模型 '{primary_name}' 状态意外变为已停止，等待中断，等待时间: {elapsed_time:.1f}秒")
                    return False, f"模型 '{primary_name}' 启动过程被意外停止"

            # 如果还在启动中，继续等待
            if current_status == ModelStatus.STARTING.value:
                time.sleep(check_interval)
            else:
                # 未知状态，记录警告并继续等待
                logger.warning(f"模型 '{primary_name}' 处于未知状态: {current_status}，继续等待...")
                time.sleep(check_interval)

    def _start_model_intelligent(self, primary_name: str) -> Tuple[bool, str]:
        """
        智能启动模型 - 包含设备检查和资源管理
        【修复】使用缓存的设备状态，防止在高并发下阻塞
        【修复】如果 Disable_GPU_monitoring 为 True，则强制启动，不依赖设备在线状态
        【适配】Linux路径兼容性处理

        Args:
            primary_name: 模型主名称

        Returns:
            (成功状态, 消息)
        """
        state = self.models_state[primary_name]
        
        # 使用 try-catch-finally 结构确保状态一致性
        try:
            # 获取自适应配置
            if self.config_manager.is_gpu_monitoring_disabled():
                # 【修复】如果禁用了GPU监控，强制假设所有所需设备在线
                logger.info("GPU监控已禁用，忽略设备在线状态，强制获取最佳配置...")
                online_devices = set()
                base_config = self.config_manager.get_model_config(primary_name)
                if base_config:
                    for key, val in base_config.items():
                        if isinstance(val, dict) and "required_devices" in val:
                            online_devices.update(val["required_devices"])
                # 如果没有获取到设备（配置可能有误），尝试使用所有已知的插件名称作为 fallback
                if not online_devices:
                    online_devices = set(self.plugin_manager.get_all_device_plugins().keys())
            else:
                # 【关键修改】获取当前在线的设备 (使用缓存，避免阻塞)
                online_devices = self.plugin_manager.get_cached_online_devices()

            model_config = self.config_manager.get_adaptive_model_config(primary_name, online_devices)
            if not model_config:
                error_msg = f"启动 '{primary_name}' 失败：没有适合当前设备状态 {list(online_devices)} 的配置方案"
                with state['lock']:
                    state['status'] = ModelStatus.FAILED.value
                    state['failure_reason'] = error_msg
                logger.error(error_msg)
                return False, error_msg

            state['current_config'] = model_config

            # 更新状态为启动脚本阶段
            with state['lock']:
                state['status'] = ModelStatus.INIT_SCRIPT.value

            # 检查设备资源
            if not self._check_and_free_resources(model_config):
                error_msg = f"启动 '{primary_name}' 失败：设备资源不足"
                with state['lock']:
                    state['status'] = ModelStatus.FAILED.value
                    state['failure_reason'] = error_msg
                logger.error(error_msg)
                return False, error_msg

            # 启动模型
            logger.info(f"正在启动模型: {primary_name} (配置方案: {model_config.get('config_source', '默认')})")
            logger.info(f"启动脚本: {model_config['script_path']}")

            # 使用进程管理器启动模型进程
            # 跨平台路径处理：使用绝对路径作为 CWD
            project_root = os.path.dirname(os.path.abspath(self.config_manager.config_path))
            process_name = f"model_{primary_name}"

            # 定义输出回调函数，将进程输出转发到日志管理器
            def output_callback(stream_type: str, message: str):
                """进程输出回调函数"""
                # 记录模型控制台输出
                self.log_manager.add_console_log(primary_name, message)

            # 跨平台进程创建标志
            creation_flags = None
            if os.name == 'nt':
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

            success, message, pid = self.process_manager.start_process(
                name=process_name,
                # 【修改】使用 script_path
                command=model_config['script_path'],
                cwd=project_root,
                description=f"模型进程: {primary_name}",
                shell=True,
                creation_flags=creation_flags,
                capture_output=True,
                output_callback=output_callback
            )

            if not success:
                error_msg = f"启动模型进程失败: {message}"
                with state['lock']:
                    state['status'] = ModelStatus.FAILED.value
                    state['failure_reason'] = error_msg
                logger.error(error_msg)
                return False, error_msg

            state.update({
                "process": None,
                "pid": pid,
                "log_thread": None,
                "stdout_thread": None,
                "stderr_thread": None
            })

            # 执行健康检查
            return self._perform_health_checks(primary_name, model_config)

        except Exception as e:
            logger.error(f"智能启动过程发生异常: {e}")
            with state['lock']:
                state['status'] = ModelStatus.FAILED.value
                state['failure_reason'] = str(e)
            return False, f"启动过程异常: {e}"

    def _check_and_free_resources(self, model_config: Dict[str, Any]) -> bool:
        """
        [优化版 - 问题三] 检查并循环释放设备资源，直到满足条件。
        """
        # 如果禁用了GPU监控，跳过所有资源检查
        if self.config_manager.is_gpu_monitoring_disabled():
            logger.info("GPU监控已禁用，跳过资源检查与释放")
            return True

        # 使用 while 循环持续尝试，直到资源足够或无法再释放
        while True:
            resource_ok = True
            deficit_devices = {}
            device_status_map = self.plugin_manager.get_device_status_snapshot()
            required_memory = model_config.get("memory_mb", {})

            # 检查每个设备的内存
            for device_name, required_mb in required_memory.items():
                device_status = device_status_map.get(device_name)
                if not device_status or not device_status.get('online', False):
                    logger.warning(f"所需设备 '{device_name}' 不在线或无信息")
                    return False # 如果依赖的设备不在线，直接失败

                device_info = device_status.get('info')
                if device_info:
                    available_mb = device_info.get('available_memory_mb', 0)
                    if available_mb < required_mb:
                        deficit_devices[device_name] = required_mb - available_mb
                        resource_ok = False
                else:
                    logger.warning(f"无法获取设备 '{device_name}' 的内存信息")
                    resource_ok = False

            # 如果资源检查通过，成功退出循环
            if resource_ok:
                logger.info("设备资源检查通过")
                return True

            logger.warning(f"设备资源不足，需要释放: {deficit_devices}")

            # 尝试停止一个空闲模型来释放资源
            freed_one_model = self._stop_idle_models_for_resources(deficit_devices)

            if freed_one_model:
                logger.info("已成功停止一个模型")
                continue
            else:
                # 如果无法再释放任何模型，但资源仍然不足，则最终失败
                logger.error("已尝试关闭所有相关空闲模型，但资源仍然不足。")
                break # 退出 while 循环

        return False

    def _stop_idle_models_for_resources(self, deficit_devices: Dict[str, int]) -> bool:
        """
        [最终优化版] 停止一个空闲模型以释放资源。
        综合了问题一、二的解决方案。
        """
        idle_candidates = []
        now = time.time()

        for name, state in self.models_state.items():
            with state['lock']:
                if state['status'] != ModelStatus.ROUTING.value:
                    continue

                # [问题一] 检查是否有待处理请求
                if self.api_router and self.api_router.pending_requests.get(name, 0) > 0:
                    logger.debug(f"跳过模型 {name}: 有 {self.api_router.pending_requests.get(name, 0)} 个待处理请求")
                    continue
                
                current_config = state.get('current_config')
                if not current_config:
                    continue
                
                used_devices = set(current_config.get('required_devices', []))
                if not used_devices:
                    used_devices = set(current_config.get('memory_mb', {}).keys())
                
                if used_devices.isdisjoint(set(deficit_devices.keys())):
                    continue
                
                last_access = state.get('last_access') or 0
                idle_seconds = max(0, now - last_access)
                
                total_memory_mb = sum(current_config.get('memory_mb', {}).values())

                # [问题二] 显存单位转换为GB并设置下限
                memory_gb = total_memory_mb / 1024.0
                memory_gb_for_score = max(0.5, memory_gb)

                # 计算淘汰评分
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

        # 只关闭分数最高的那个模型
        candidate_to_stop = sorted_candidates[0]
        model_name = candidate_to_stop['name']
        logger.info(f"为释放资源，正在停止模型: {model_name} (当前评分最高)")
        success, message = self.stop_model(model_name)

        if success:
            logger.info(f"模型 {model_name} 已成功停止")
            return True # 返回True表示成功关闭了一个
        else:
            logger.warning(f"尝试停止模型 {model_name} 失败: {message}")
            return False

    def _perform_health_checks(self, primary_name: str, model_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        执行健康检查

        Args:
            primary_name: 模型主名称
            model_config: 模型配置

        Returns:
            (成功状态, 消息)
        """
        state = self.models_state[primary_name]
        port = model_config['port']
        timeout_seconds = 300
        start_time = time.time()

        # 健康检查
        with state['lock']:
            state['status'] = ModelStatus.HEALTH_CHECK.value

        logger.info(f"正在对模型 '{primary_name}' 进行健康检查")

        model_mode = model_config.get("mode", "Chat")
        interface_plugin = self.plugin_manager.get_interface_plugin(model_mode)

        if interface_plugin:
            # 使用插件的统一健康检查方法
            health_success, health_message = interface_plugin.health_check(primary_name, port, start_time, timeout_seconds)
            if health_success:
                logger.info(f"模型 '{primary_name}' 健康检查通过")
                with state['lock']:
                    state['status'] = ModelStatus.ROUTING.value
                    state['last_access'] = time.time()

                # 记录模型启动时间并启动定时监控
                self.runtime_monitor.record_model_start(primary_name)

                return True, f"模型 {primary_name} 启动成功"
            else:
                logger.error(f"模型 '{primary_name}' 健康检查失败: {health_message}")
                state['status'] = ModelStatus.FAILED.value
                state['failure_reason'] = health_message
                self.stop_model(primary_name)
                return False, f"健康检查失败: {health_message}"
        else:
            msg = f"未找到模式 '{model_mode}' 的接口插件，无法进行健康检查"
            logger.error(msg)
            state['status'] = ModelStatus.FAILED.value
            state['failure_reason'] = msg
            self.stop_model(primary_name)
            return False, msg

    def stop_model(self, primary_name: str) -> Tuple[bool, str]:
        """
        停止模型 - 接受主名称，增强支持中断正在启动的模型

        Args:
            primary_name: 模型主名称

        Returns:
            (成功状态, 消息)
        """
        state = self.models_state[primary_name]
        model_lock = self.startup_locks[primary_name]

        # 【新增】检查启动锁，处理正在启动的模型
        startup_lock_released = False
        try:
            # 尝试获取启动锁，如果获取失败说明模型正在启动
            startup_lock_released = model_lock.acquire(blocking=False)
            if not startup_lock_released:
                logger.info(f"模型 '{primary_name}' 正在启动中，尝试中断启动过程...")
                # 模型正在启动，尝试获取锁来中断启动
                # 等待最多10秒来获取启动锁
                startup_lock_released = model_lock.acquire(blocking=True, timeout=10)
                if not startup_lock_released:
                    logger.warning(f"无法获取模型 '{primary_name}' 的启动锁，可能存在死锁")
                    # 强制释放锁作为最后手段
                    logger.warning(f"强制释放模型 '{primary_name}' 的启动锁...")
                    # 这里我们不能直接强制释放锁，但可以标记状态为停止
                    with state['lock']:
                        state['status'] = ModelStatus.STOPPED.value
                        state['failure_reason'] = "被外部请求中断"
                    return False, f"无法中断模型 '{primary_name}' 的启动过程，请稍后重试"
                else:
                    logger.info(f"成功获取模型 '{primary_name}' 的启动锁，将中断启动过程")
        except Exception as e:
            logger.error(f"处理模型 '{primary_name}' 启动锁时发生异常: {e}")

        try:
            with state['lock']:
                # 检查当前状态
                if state['status'] in [ModelStatus.STOPPED.value, ModelStatus.FAILED.value]:
                    logger.info(f"模型 '{primary_name}' 已停止或失败")
                    return True, f"模型 '{primary_name}' 已停止或失败"

                # 标记为停止状态，阻止启动过程继续
                original_status = state['status']
                state['status'] = ModelStatus.STOPPED.value
                state['failure_reason'] = "被用户请求停止"

                logger.info(f"正在停止模型 '{primary_name}' (当前状态: {original_status})")

                pid = state.get('pid')
                if pid:
                    logger.info(f"正在强制停止模型 {primary_name} (PID: {pid})")
                    # 使用进程管理器强制终止模型进程
                    process_name = f"model_{primary_name}"
                    success, message = self.process_manager.stop_process(process_name, force=True)

                    if success:
                        logger.info(f"模型 {primary_name} (PID: {pid}) 已成功终止")
                    else:
                        logger.warning(f"终止模型 {primary_name} PID:{pid} 失败: {message}")
                else:
                    logger.info(f"模型 '{primary_name}' 没有关联的进程，直接标记为停止")

                self._mark_model_as_stopped(primary_name, acquire_lock=False)

            # 只有在模型被监控时才记录停止时间
            if self.runtime_monitor.is_model_monitored(primary_name):
                self.runtime_monitor.record_model_stop(primary_name)

            # 停止后刷新设备状态缓存
            logger.info(f"模型 '{primary_name}' 已停止，正在刷新设备状态缓存...")
            self.plugin_manager.update_device_status()

            return True, f"模型 {primary_name} 已停止"
        finally:
            # 【重要】释放启动锁
            if startup_lock_released:
                try:
                    model_lock.release()
                    logger.info(f"已释放模型 '{primary_name}' 的启动锁")
                except Exception as e:
                    logger.error(f"释放模型 '{primary_name}' 启动锁时发生异常: {e}")

    def _mark_model_as_stopped(self, primary_name: str, acquire_lock: bool = True):
        """
        标记模型为已停止状态

        Args:
            primary_name: 模型主名称
            acquire_lock: 是否获取锁
        """
        state = self.models_state[primary_name]

        def update():
            state.update({
                "process": None,
                "pid": None,
                "status": ModelStatus.STOPPED.value,
                "last_access": None,
                "log_thread": None,
                "current_config": None,
                "failure_reason": None
            })

        if acquire_lock:
            with state['lock']:
                update()
        else:
            update()

    def unload_all_models(self):
        """
        卸载所有运行中的模型
        [重构] 基于 stop_model 实现，避免代码重复，自动获得缓存刷新功能
        """
        logger.info("正在卸载所有运行中的模型...")
        primary_names = list(self.models_state.keys())

        terminated_count = 0
        failed_models = []

        # [优化] 直接调用 stop_model，复用其所有逻辑（包括缓存刷新）
        for name in primary_names:
            try:
                success, message = self.stop_model(name)
                if success:
                    terminated_count += 1
                    logger.debug(f"模型 '{name}' 已成功卸载")
                else:
                    logger.warning(f"模型 '{name}' 卸载失败: {message}")
                    failed_models.append(name)
            except Exception as e:
                logger.error(f"卸载模型 '{name}' 时发生异常: {e}")
                failed_models.append(name)

        logger.info(f"所有模型卸载完成，成功: {terminated_count}/{len(primary_names)}")
        if failed_models:
            logger.warning(f"以下模型卸载失败: {failed_models}")

        return terminated_count

    def idle_check_loop(self):
        """
        【已修复】空闲检查循环 - 解决死锁问题并增加活跃请求检查
        逻辑：只有当模型超时且当前没有正在处理的请求时，才执行关闭
        """
        while self.is_running:
            time.sleep(30)
            if not self.is_running:
                break

            try:
                alive_time_min = self.config_manager.get_alive_time()
                # 如果配置为 0 或负数，代表永不自动关闭
                if alive_time_min <= 0:
                    continue

                alive_time_sec = alive_time_min * 60
                now = time.time()

                # 收集需要停止的模型列表
                models_to_stop = []

                for name in list(self.models_state.keys()):
                    state = self.models_state[name]
                    # 只在锁内读取状态
                    with state['lock']:
                        # 只有处于 ROUTING 状态的模型才需要检查空闲
                        if state['status'] != ModelStatus.ROUTING.value:
                            continue
                            
                        last_access = state.get('last_access')
                        if not last_access:
                            continue

                        # 【初步检查】获取待处理请求数
                        pending_count = 0
                        if self.api_router:
                            pending_count = self.api_router.pending_requests.get(name, 0)

                        # 计算空闲时间
                        idle_duration = now - last_access

                        # 条件1: 没有任何正在处理的请求
                        # 条件2: 空闲时间超过了设定阈值
                        if pending_count == 0 and idle_duration > alive_time_sec:
                            # logger.info(f"模型 {name} 判定为空闲: 无请求且已空闲 {idle_duration:.1f}秒")
                            models_to_stop.append(name)
                
                # 在锁外执行停止操作，避免死锁
                for name in models_to_stop:
                    # 【双重检查】防止在“决定关闭”和“执行关闭”的微小时间窗内有新请求进入
                    should_stop = True
                    if self.api_router:
                        current_pending = self.api_router.pending_requests.get(name, 0)
                        if current_pending > 0:
                            logger.info(f"模型 {name} 在关闭前一刻收到新请求，取消关闭")
                            should_stop = False
                    
                    if should_stop:
                        logger.info(f"执行自动关闭策略: 停止模型 {name}")
                        self.stop_model(name)

            except Exception as e:
                logger.error(f"空闲检查线程出错: {e}", exc_info=True)

    def get_all_models_status(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有模型状态

        Returns:
            模型状态字典
        """
        status_copy = {}
        now = time.time()

        # 【关键修改】使用缓存的设备状态，避免阻塞
        online_devices = self.plugin_manager.get_cached_online_devices()

        for primary_name, state in self.models_state.items():
            idle_seconds = (now - state['last_access']) if state.get('last_access') else -1
            config = self.config_manager.get_model_config(primary_name)

            # 计算自适应配置
            adaptive_config = self.config_manager.get_adaptive_model_config(primary_name, online_devices)

            status_copy[primary_name] = {
                "aliases": config.get("aliases", [primary_name]) if config else [primary_name],
                "status": state['status'],
                "pid": state['pid'],
                "idle_time_sec": f"{idle_seconds:.0f}" if idle_seconds != -1 else "N/A",
                "mode": config.get("mode", "Chat") if config else "Chat",
                "is_available": bool(adaptive_config),
                # 【修改】current_bat_path -> current_script_path, 读取 script_path
                "current_script_path": adaptive_config.get("script_path", "") if adaptive_config else "无可用配置",
                "config_source": adaptive_config.get("config_source", "N/A") if adaptive_config else "N/A",
                "failure_reason": state.get('failure_reason')
            }

        return status_copy

    
    def get_model_logs(self, primary_name: str) -> List[Dict[str, Any]]:
        """
        获取模型的控制台日志 - 接受主名称

        Args:
            primary_name: 模型主名称

        Returns:
            模型控制台日志列表
        """
        try:
            return self.log_manager.get_logs(primary_name)
        except Exception as e:
            logger.error(f"获取模型日志失败: {e}")
            return [{"timestamp": time.time(), "message": f"获取模型控制台日志失败: {str(e)}"}]

    def subscribe_to_model_logs(self, primary_name: str) -> queue.Queue:
        """
        订阅模型控制台日志流

        Args:
            primary_name: 模型主名称

        Returns:
            订阅队列
        """
        return self.log_manager.subscribe_to_logs(primary_name)

    def unsubscribe_from_model_logs(self, primary_name: str, subscriber_queue: queue.Queue):
        """
        取消订阅模型控制台日志流

        Args:
            primary_name: 模型主名称
            subscriber_queue: 订阅队列
        """
        self.log_manager.unsubscribe_from_logs(primary_name, subscriber_queue)

    def get_log_stats(self) -> Dict[str, Any]:
        """
        获取模型控制台日志统计信息

        Returns:
            统计信息字典
        """
        return self.log_manager.get_log_stats()

    def get_model_list(self) -> Dict[str, Any]:
        """
        获取模型列表

        Returns:
            模型列表字典
        """
        data = []
        for primary_name in self.models_state.keys():
            config = self.config_manager.get_model_config(primary_name)
            if config:
                data.append({
                    "id": primary_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "user",
                    "aliases": config.get("aliases", []),
                    "mode": config.get("mode", "Chat")
                })

        return {"object": "list", "data": data}

    
    
    def shutdown(self):
        """优化的关闭模型控制器 - 并行终止所有进程"""
        logger.info("正在快速关闭模型控制器...")
        self.is_running = False
        self.shutdown_event.set()
        
        # 停止插件监控线程
        if self.plugin_manager:
            self.plugin_manager.stop_monitor()

        # 取消所有未完成的启动任务
        for model_name, future in self.startup_futures.items():
            if not future.done():
                future.cancel()
                logger.info(f"取消模型 {model_name} 的启动任务")

        # 使用进程管理器并行终止所有模型进程
        logger.info("正在并行终止所有模型进程...")
        terminated_models = []

        # 构建停止任务列表
        stop_tasks = []
        for primary_name, state in self.models_state.items():
            with state['lock']:
                pid = state.get('pid')
                if pid:
                    process_name = f"model_{primary_name}"
                    stop_tasks.append((primary_name, process_name))

        # 并行停止进程
        def stop_model_task(primary_name, process_name):
            try:
                success, message = self.process_manager.stop_process(process_name, force=True, timeout=3)
                if success:
                    return primary_name, True, None
                else:
                    return primary_name, False, message
            except Exception as e:
                return primary_name, False, str(e)

        # 提交停止任务
        stop_futures = []
        for primary_name, process_name in stop_tasks:
            future = self.executor.submit(stop_model_task, primary_name, process_name)
            stop_futures.append(future)

        # 收集结果，设置超时
        timeout = len(stop_tasks) * 2 + 5  # 总超时时间
        try:
            for future in concurrent.futures.as_completed(stop_futures, timeout=timeout):
                try:
                    primary_name, success, message = future.result()
                    if success:
                        terminated_models.append(primary_name)

                        # 更新状态
                        state = self.models_state[primary_name]
                        with state['lock']:
                            state['pid'] = None
                            state['status'] = ModelStatus.STOPPED.value
                            state['process'] = None
                    else:
                        logger.warning(f"停止模型 {primary_name} 失败: {message}")
                except Exception as e:
                    logger.error(f"处理模型停止结果时发生异常: {e}")
        except concurrent.futures.TimeoutError:
            logger.error("模型停止超时")

        # 只记录正在运行的模型的停止时间
        for primary_name, state in self.models_state.items():
            if self.runtime_monitor.is_model_monitored(primary_name):
                self.runtime_monitor.record_model_stop(primary_name)

        # 关闭运行时间监控器
        self.runtime_monitor.shutdown()

        # 关闭日志管理器
        self.log_manager.shutdown()

        # 关闭线程池
        self.executor.shutdown(wait=True)

        logger.info(f"模型控制器关闭完成，已终止 {len(terminated_models)} 个模型进程")
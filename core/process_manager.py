"""
统一的进程管理器 (Windows/Linux 兼容版)
提供进程启动、追踪、输出捕获和跨平台终止功能
"""

import subprocess
import time
import threading
import psutil
import concurrent.futures
import os
from typing import Dict, Optional, Tuple, List, Any, Callable
from dataclasses import dataclass
from enum import Enum
from utils.logger import get_logger

logger = get_logger(__name__)


class ProcessStatus(Enum):
    """进程状态枚举"""
    STOPPED = "stopped"
    RUNNING = "running"
    STARTING = "starting"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass
class ProcessInfo:
    """进程信息"""
    pid: int
    name: str
    status: ProcessStatus
    process: Optional[subprocess.Popen] = None
    start_time: Optional[float] = None
    stop_time: Optional[float] = None
    exit_code: Optional[int] = None
    command: Optional[str] = None
    description: Optional[str] = None
    output_callback: Optional[Callable[[str, str], None]] = None
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None


class ProcessManager:
    """跨平台统一进程管理器"""

    def __init__(self):
        """初始化进程管理器"""
        self.processes: Dict[str, ProcessInfo] = {}
        self.lock = threading.RLock()  # 使用RLock支持重入
        self.monitor_thread = None
        self.is_monitoring = True
        self.shutdown_event = threading.Event()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self._process_cleanup_complete = threading.Event()

        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_processes, daemon=True)
        self.monitor_thread.start()

        logger.info(f"进程管理器初始化完成 (OS: {os.name})")

    def start_process(
        self,
        name: str,
        command: str,
        cwd: Optional[str] = None,
        description: Optional[str] = None,
        shell: bool = True,
        creation_flags: Optional[int] = None,
        capture_output: bool = False,
        output_callback: Optional[Callable[[str, str], None]] = None
    ) -> Tuple[bool, str, Optional[int]]:
        """
        启动进程 (跨平台适配)
        """
        with self.lock:
            # 检查是否已存在同名进程
            if name in self.processes:
                existing = self.processes[name]
                if existing.status == ProcessStatus.RUNNING:
                    return False, f"进程 '{name}' 已在运行", existing.pid
                elif existing.status == ProcessStatus.STARTING:
                    return False, f"进程 '{name}' 正在启动中", existing.pid

            # 创建进程信息
            process_info = ProcessInfo(
                pid=0,
                name=name,
                status=ProcessStatus.STARTING,
                command=command,
                description=description,
                output_callback=output_callback
            )

            self.processes[name] = process_info

        try:
            logger.info(f"正在启动进程: {name} - {command}")

            # 准备启动参数
            startup_params = {
                'shell': shell,
                'text': True,
                'encoding': 'utf-8',
                'errors': 'replace'
            }

            if cwd:
                startup_params['cwd'] = cwd

            # --- 跨平台参数适配 ---
            if os.name == 'nt':
                # Windows 特定参数
                if creation_flags:
                    startup_params['creationflags'] = creation_flags
            else:
                # Linux/Unix 特定参数
                # close_fds=True 避免子进程继承不必要的文件描述符
                startup_params['close_fds'] = True
            # ----------------------

            if capture_output:
                startup_params.update({
                    'stdout': subprocess.PIPE,
                    'stderr': subprocess.PIPE,
                    'bufsize': 1
                })

            # 启动进程
            process = subprocess.Popen(command, **startup_params)

            # 更新进程信息
            with self.lock:
                process_info.pid = process.pid
                process_info.process = process
                process_info.start_time = time.time()
                process_info.status = ProcessStatus.RUNNING

            # 如果需要捕获输出，启动输出监控线程
            if capture_output and output_callback:
                process_info.stdout_thread = threading.Thread(
                    target=self._monitor_output,
                    args=(process, process_info, 'stdout', output_callback),
                    daemon=True
                )
                process_info.stderr_thread = threading.Thread(
                    target=self._monitor_output,
                    args=(process, process_info, 'stderr', output_callback),
                    daemon=True
                )
                process_info.stdout_thread.start()
                process_info.stderr_thread.start()

            logger.info(f"进程启动成功: {name} (PID: {process.pid})")
            return True, f"进程启动成功", process.pid

        except Exception as e:
            with self.lock:
                process_info.status = ProcessStatus.FAILED
                process_info.stop_time = time.time()

            logger.error(f"启动进程失败: {name} - {e}")
            return False, f"启动进程失败: {e}", None

    def _monitor_output(self, process: subprocess.Popen, process_info: ProcessInfo, stream_type: str, callback: Callable[[str, str], None]):
        """监控进程输出流"""
        stream = getattr(process, stream_type)

        try:
            for line in iter(stream.readline, ''):
                if line:
                    line = line.rstrip('\n\r')
                    if line.strip():  # 只处理非空行
                        callback(stream_type, line)
                else:
                    break
        except Exception as e:
            # 进程退出时可能会触发读错误，忽略
            pass
        finally:
            try:
                stream.close()
            except:
                pass

    def stop_process(
        self,
        name: str,
        force: bool = False,
        timeout: int = 10
    ) -> Tuple[bool, str]:
        """
        停止进程
        """
        with self.lock:
            if name not in self.processes:
                return True, f"进程 '{name}' 不存在"

            process_info = self.processes[name]

            if process_info.status == ProcessStatus.STOPPED:
                return True, f"进程 '{name}' 已停止"

            if process_info.status == ProcessStatus.STOPPING:
                return True, f"进程 '{name}' 正在停止中"

            # 标记为停止中
            process_info.status = ProcessStatus.STOPPING

        try:
            pid = process_info.pid
            logger.info(f"正在停止进程: {name} (PID: {pid}, 强制: {force})")

            if force:
                # 强制终止进程树
                success = self._kill_process_tree(pid)
                if success:
                    logger.info(f"强制终止进程成功: {name} (PID: {pid})")
                    message = f"进程已强制终止"
                else:
                    logger.warning(f"强制终止进程失败: {name} (PID: {pid})")
                    message = f"强制终止进程失败"
            else:
                # 正常关闭
                success = self._terminate_process(pid, timeout)
                if success:
                    logger.info(f"正常关闭进程成功: {name} (PID: {pid})")
                    message = f"进程已正常关闭"
                else:
                    logger.warning(f"正常关闭进程失败: {name} (PID: {pid})")
                    message = f"正常关闭进程失败"

            # 更新进程状态并清理资源
            with self.lock:
                process_info.status = ProcessStatus.STOPPED
                process_info.stop_time = time.time()
                process_info.process = None

                # 清理输出监控线程引用
                process_info.stdout_thread = None
                process_info.stderr_thread = None

            return success, message

        except Exception as e:
            logger.error(f"停止进程失败: {name} - {e}")
            return False, f"停止进程失败: {e}"

    def _terminate_process(self, pid: int, timeout: int) -> bool:
        """优化的正常终止进程 (跨平台)"""
        try:
            # 尝试正常终止
            process = psutil.Process(pid)
            process.terminate()  # Windows: TerminateProcess, Linux: SIGTERM

            # 减少等待时间，提高性能
            try:
                process.wait(timeout=min(timeout, 5))  # 最多等待5秒
                return True
            except psutil.TimeoutExpired:
                logger.warning(f"进程 {pid} 超时未结束，尝试强制终止")
                return self._kill_process_tree(pid)

        except psutil.NoSuchProcess:
            # 进程已不存在
            return True
        except Exception as e:
            logger.error(f"终止进程 {pid} 失败: {e}")
            return False

    def _kill_process_tree(self, pid: int) -> bool:
        """强制终止进程树 (跨平台兼容)"""
        # 1. 首先尝试 psutil (跨平台最通用方案)
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            # 终止子进程
            for child in children:
                try:
                    child.kill() # Linux: SIGKILL
                except Exception:
                    pass

            # 终止主进程
            parent.kill()
            
            # 简单等待确认
            _, alive = psutil.wait_procs([parent] + children, timeout=3)
            if not alive:
                logger.info(f"psutil成功终止进程树: {pid}")
                return True

        except (psutil.NoSuchProcess, Exception):
            pass

        # 2. 备选方案：使用系统命令
        if os.name == 'nt':
            # Windows 使用 taskkill
            try:
                result = subprocess.run(
                    ['taskkill', '/F', '/T', '/PID', str(pid)],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    timeout=3
                )
                return result.returncode == 0 or "没有找到" in result.stderr
            except Exception:
                return False
        else:
            # Linux 使用 kill -9
            try:
                # 尝试强制杀死主进程 (子进程可能变成孤儿进程被init回收，或者由于pgid被清理)
                result = subprocess.run(
                    ['kill', '-9', str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                return result.returncode == 0
            except Exception:
                return False

    def _is_process_alive(self, pid: int) -> bool:
        """检查进程是否存活"""
        try:
            process = psutil.Process(pid)
            if process.status() == psutil.STATUS_ZOMBIE:
                return False
            return process.is_running()
        except psutil.NoSuchProcess:
            return False
        except Exception as e:
            return False

    def _monitor_processes(self):
        """优化的进程监控状态"""
        while self.is_monitoring:
            try:
                if self.shutdown_event.is_set():
                    break

                time.sleep(10)  # 改为每10秒检查一次

                with self.lock:
                    dead_processes = []
                    
                    # 使用 list() 复制 keys，防止迭代时修改字典
                    for name, process_info in list(self.processes.items()):
                        if process_info.status in [ProcessStatus.RUNNING, ProcessStatus.STARTING]:
                            # 检查进程是否存活
                            if not self._is_process_alive(process_info.pid):
                                logger.info(f"检测到进程已退出: {name} (PID: {process_info.pid})")
                                process_info.status = ProcessStatus.STOPPED
                                process_info.stop_time = time.time()
                                process_info.process = None

                                # 尝试获取退出码
                                try:
                                    if process_info.process:
                                        process_info.exit_code = process_info.process.poll()
                                except Exception:
                                    pass

                    # 优化清理策略 - 只保留最近50个已停止进程
                    stopped_count = sum(1 for p in self.processes.values()
                                     if p.status == ProcessStatus.STOPPED)
                    if stopped_count > 50:
                        # 按停止时间排序，删除最旧的
                        stopped_processes = [(name, p) for name, p in self.processes.items()
                                          if p.status == ProcessStatus.STOPPED]
                        stopped_processes.sort(key=lambda x: x[1].stop_time or 0)

                        for name, _ in stopped_processes[:stopped_count - 50]:
                            del self.processes[name]
                            logger.debug(f"清理已停止进程记录: {name}")

            except Exception as e:
                logger.error(f"进程监控线程出错: {e}")
                if self.shutdown_event.is_set():
                    break

    def get_process_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取进程信息"""
        with self.lock:
            if name not in self.processes:
                return None

            process_info = self.processes[name]

            # 如果进程正在运行，更新实时状态
            if process_info.status in [ProcessStatus.RUNNING, ProcessStatus.STARTING]:
                is_alive = self._is_process_alive(process_info.pid)
                if not is_alive:
                    process_info.status = ProcessStatus.STOPPED
                    process_info.stop_time = time.time()

            return {
                'name': process_info.name,
                'pid': process_info.pid,
                'status': process_info.status.value,
                'start_time': process_info.start_time,
                'stop_time': process_info.stop_time,
                'exit_code': process_info.exit_code,
                'command': process_info.command,
                'description': process_info.description,
                'uptime': time.time() - process_info.start_time if process_info.start_time else 0
            }

    def list_processes(self) -> List[Dict[str, Any]]:
        """列出所有进程"""
        with self.lock:
            processes = []
            for name in self.processes:
                info = self.get_process_info(name)
                if info:
                    processes.append(info)
            return processes

    def stop_all_processes(self, force: bool = False) -> Dict[str, Tuple[bool, str]]:
        """优化的并行停止所有进程"""
        results = {}

        with self.lock:
            process_names = list(self.processes.keys())

        if not process_names:
            self._process_cleanup_complete.set()
            return results

        logger.info(f"并行停止 {len(process_names)} 个进程...")

        # 使用线程池并行停止进程
        def stop_single_process(name):
            success, message = self.stop_process(name, force, timeout=3 if force else 5)
            return name, (success, message)

        # 提交所有停止任务
        future_to_name = {}
        for name in process_names:
            future = self.executor.submit(stop_single_process, name)
            future_to_name[future] = name

        # 收集结果，设置超时
        timeout = len(process_names) * 2  # 总超时时间
        try:
            for future in concurrent.futures.as_completed(future_to_name, timeout=timeout):
                name = future_to_name[future]
                try:
                    name, result = future.result()
                    results[name] = result
                except Exception as e:
                    logger.error(f"停止进程 {name} 时发生异常: {e}")
                    results[name] = (False, f"停止进程异常: {e}")
        except concurrent.futures.TimeoutError:
            logger.error("停止所有进程超时")
            # 处理未完成的进程
            for future, name in future_to_name.items():
                if not future.done():
                    future.cancel()
                    results[name] = (False, "停止进程超时")

        self._process_cleanup_complete.set()
        logger.info(f"进程停止完成，成功: {sum(1 for r in results.values() if r[0])}/{len(results)}")
        return results

    def cleanup(self):
        """优化的清理进程管理器"""
        logger.info("正在清理进程管理器...")
        self.is_monitoring = False
        self.shutdown_event.set()

        # 停止所有进程
        self.stop_all_processes(force=True)

        # 等待进程清理完成或超时
        if not self._process_cleanup_complete.wait(timeout=15):
            logger.warning("进程清理超时，强制退出")

        # 关闭线程池
        self.executor.shutdown(wait=True)

        # 等待监控线程结束
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=3)

        logger.info("进程管理器清理完成")

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except Exception:
            pass


# 全局进程管理器实例
_global_process_manager = None


def get_process_manager() -> ProcessManager:
    """获取全局进程管理器实例"""
    global _global_process_manager
    if _global_process_manager is None:
        _global_process_manager = ProcessManager()
    return _global_process_manager


def initialize_process_manager():
    """初始化全局进程管理器"""
    global _global_process_manager
    if _global_process_manager is not None:
        _global_process_manager.cleanup()
    _global_process_manager = ProcessManager()
    return _global_process_manager


def cleanup_process_manager():
    """清理全局进程管理器"""
    global _global_process_manager
    if _global_process_manager is not None:
        _global_process_manager.cleanup()
        _global_process_manager = None
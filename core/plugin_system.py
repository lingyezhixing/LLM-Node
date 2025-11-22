"""
LLM-Manager 插件系统
合并的插件加载和管理功能，支持即插即用
"""
import os
import sys
import importlib.util
import inspect
import logging
import time
import threading
from typing import Dict, List, Type, Any, Optional
from abc import ABC

logger = logging.getLogger(__name__)

class PluginLoader:
    """插件加载器基类"""

    def __init__(self, plugin_dir: str, base_class: Type[ABC]):
        self.plugin_dir = plugin_dir
        self.base_class = base_class
        self.loaded_plugins: Dict[str, Any] = {}

    def discover_plugins(self) -> Dict[str, Type[ABC]]:
        """
        自动发现插件目录中的所有插件类
        返回: {插件名称: 插件类}
        """
        plugins = {}

        if not os.path.exists(self.plugin_dir):
            logger.warning(f"插件目录不存在: {self.plugin_dir}")
            return plugins

        # 遍历插件目录
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith('.py') and not filename.startswith('__') and filename != 'Base_Class.py':
                plugin_name = filename[:-3]  # 移除.py后缀
                plugin_path = os.path.join(self.plugin_dir, filename)

                try:
                    plugin_class = self._load_plugin_from_file(plugin_path, plugin_name)
                    if plugin_class:
                        plugins[plugin_name] = plugin_class
                        logger.debug(f"发现插件: {plugin_name}")
                except Exception as e:
                    logger.error(f"加载插件文件失败 {filename}: {e}")

        return plugins

    def _load_plugin_from_file(self, file_path: str, plugin_name: str) -> Optional[Type[ABC]]:
        """从Python文件中加载插件类"""
        try:
            # 获取插件目录的绝对路径
            plugin_dir = os.path.abspath(os.path.dirname(self.plugin_dir))
            if not plugin_dir in sys.path:
                sys.path.insert(0, plugin_dir)

            # 动态导入模块
            spec = importlib.util.spec_from_file_location(plugin_name, file_path)
            if not spec or not spec.loader:
                logger.error(f"无法创建模块规范: {file_path}")
                return None

            module = importlib.util.module_from_spec(spec)

            # 设置模块的__package__以支持相对导入
            module.__package__ = os.path.basename(os.path.dirname(file_path))

            spec.loader.exec_module(module)

            # 查找插件类
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, self.base_class) and
                    obj != self.base_class):

                    # 验证插件类是否有必要的抽象方法
                    self._validate_plugin_class(obj)
                    return obj

            logger.warning(f"在文件 {file_path} 中未找到有效的插件类")
            return None

        except Exception as e:
            logger.error(f"加载插件文件失败 {file_path}: {e}")
            return None

    def _validate_plugin_class(self, plugin_class: Type[ABC]) -> bool:
        """验证插件类是否实现了所有必需的抽象方法"""
        abstract_methods = self.base_class.__abstractmethods__
        for method_name in abstract_methods:
            if not hasattr(plugin_class, method_name):
                raise ValueError(f"插件类 {plugin_class.__name__} 缺少必需的抽象方法: {method_name}")

        return True

    def load_plugins(self, **kwargs) -> Dict[str, Any]:
        """
        加载所有发现的插件并创建实例
        返回: {插件名称: 插件实例}
        """
        discovered_plugins = self.discover_plugins()

        for plugin_name, plugin_class in discovered_plugins.items():
            try:
                # 创建插件实例
                if hasattr(plugin_class, '__init__'):
                    # 获取构造函数参数
                    sig = inspect.signature(plugin_class.__init__)
                    params = {}

                    # 为接口插件传递model_manager
                    if hasattr(self, 'model_manager') and self.model_manager:
                        params['model_manager'] = self.model_manager

                    # 传递其他关键字参数
                    for param_name, param in sig.parameters.items():
                        if param_name != 'self' and param_name in kwargs:
                            params[param_name] = kwargs[param_name]

                    plugin_instance = plugin_class(**params)
                else:
                    plugin_instance = plugin_class()

                # 获取插件标识符
                plugin_id = self._get_plugin_id(plugin_instance)
                if plugin_id:
                    self.loaded_plugins[plugin_id] = plugin_instance
                    logger.debug(f"成功加载插件: {plugin_name} -> {plugin_id}")
                else:
                    logger.warning(f"插件 {plugin_name} 没有返回有效的标识符")

            except Exception as e:
                logger.error(f"实例化插件失败 {plugin_name}: {e}")

        return self.loaded_plugins

    def _get_plugin_id(self, plugin_instance: Any) -> Optional[str]:
        """获取插件实例的标识符，子类需要重写此方法"""
        return None

    def get_plugin(self, plugin_id: str) -> Optional[Any]:
        """获取指定ID的插件实例"""
        return self.loaded_plugins.get(plugin_id)

    def get_all_plugins(self) -> Dict[str, Any]:
        """获取所有已加载的插件"""
        return self.loaded_plugins.copy()

    def reload_plugins(self, **kwargs) -> Dict[str, Any]:
        """重新加载所有插件"""
        self.loaded_plugins.clear()
        return self.load_plugins(**kwargs)

class DevicePluginLoader(PluginLoader):
    """设备插件加载器"""

    def __init__(self, plugin_dir: str = "plugins/devices"):
        from plugins.devices.Base_Class import DevicePlugin
        super().__init__(plugin_dir, DevicePlugin)

    def _get_plugin_id(self, plugin_instance: Any) -> Optional[str]:
        """设备插件使用device_name作为标识符"""
        if hasattr(plugin_instance, 'device_name'):
            return plugin_instance.device_name
        return None

class InterfacePluginLoader(PluginLoader):
    """接口插件加载器"""

    def __init__(self, plugin_dir: str = "plugins/interfaces", model_manager=None):
        from plugins.interfaces.Base_Class import InterfacePlugin
        self.model_manager = model_manager
        super().__init__(plugin_dir, InterfacePlugin)

    def load_plugins(self, **kwargs) -> Dict[str, Any]:
        """重写加载方法，传入model_manager"""
        self.model_manager = kwargs.get('model_manager', self.model_manager)
        # 移除重复的model_manager参数，只传递一次
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'model_manager'}
        return super().load_plugins(model_manager=self.model_manager, **filtered_kwargs)

    def _get_plugin_id(self, plugin_instance: Any) -> Optional[str]:
        """接口插件使用interface_name作为标识符"""
        if hasattr(plugin_instance, 'interface_name'):
            return plugin_instance.interface_name
        return None

class PluginManager:
    """统一的插件管理器"""

    def __init__(self, device_dir: str = "plugins/devices", interface_dir: str = "plugins/interfaces"):
        self.device_dir = device_dir
        self.interface_dir = interface_dir
        self.device_loader = DevicePluginLoader(device_dir)
        self.interface_loader = InterfacePluginLoader(interface_dir)
        self.device_plugins: Dict[str, Any] = {}
        self.interface_plugins: Dict[str, Any] = {}
        self.last_reload_time = 0
        
        # --- 设备状态缓存机制 (解决死锁的核心) ---
        self.device_status_cache = {}
        self.cache_lock = threading.RLock()
        self.monitor_thread = None
        self.is_monitoring = False

    def start_monitor(self):
        """启动设备状态后台监控线程"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        # 先进行一次同步更新，确保启动时缓存有数据
        self._update_device_status_once()
        
        self.monitor_thread = threading.Thread(target=self._monitor_devices_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("设备状态监控线程已启动")

    def stop_monitor(self):
        """停止设备状态监控线程"""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)

    def _monitor_devices_loop(self):
        """后台循环更新设备状态"""
        while self.is_monitoring:
            try:
                self._update_device_status_once()
            except Exception as e:
                logger.error(f"设备状态更新失败: {e}")
            
            # 休眠3秒，减少对底层驱动的压力
            for _ in range(30):
                if not self.is_monitoring:
                    break
                time.sleep(0.1)

    def _update_device_status_once(self):
        """执行一次设备状态更新"""
        new_cache = {}
        for name, plugin in self.device_plugins.items():
            try:
                # 注意：这些调用可能会阻塞，放在后台线程执行
                is_online = plugin.is_online() if hasattr(plugin, 'is_online') else False
                device_info = plugin.get_devices_info() if hasattr(plugin, 'get_devices_info') and is_online else None
                
                new_cache[name] = {
                    "online": is_online,
                    "info": device_info,
                    "type": type(plugin).__name__
                }
            except Exception as e:
                logger.warning(f"更新设备 {name} 状态时出错: {e}")
                new_cache[name] = {
                    "online": False, 
                    "error": str(e),
                    "type": type(plugin).__name__
                }

        with self.cache_lock:
            self.device_status_cache = new_cache

    def get_device_status_snapshot(self) -> Dict[str, Any]:
        """
        获取设备状态的快照（从缓存读取，非阻塞）
        用于API快速响应和模型启动前的快速检查
        """
        with self.cache_lock:
            # 返回深拷贝副本，防止外部修改
            return {k: v.copy() for k, v in self.device_status_cache.items()}

    def get_cached_online_devices(self) -> set:
        """获取当前缓存中显示的在线设备集合"""
        with self.cache_lock:
            return {name for name, data in self.device_status_cache.items() if data.get("online", False)}

    # ----------------------------------------

    def load_all_plugins(self, model_manager=None) -> Dict[str, Dict[str, Any]]:
        """加载所有插件"""
        result = {
            "device_plugins": {},
            "interface_plugins": {}
        }

        # 加载设备插件
        try:
            self.device_plugins = self.device_loader.load_plugins()
            
            # 初始填充缓存（使用默认值，等待monitor线程更新真实值）
            with self.cache_lock:
                for name, plugin in self.device_plugins.items():
                    if name not in self.device_status_cache:
                        self.device_status_cache[name] = {
                            "online": False,
                            "info": None,
                            "type": type(plugin).__name__
                        }

            result["device_plugins"] = {
                name: {
                    "status": "loaded",
                    "type": type(plugin).__name__
                }
                for name, plugin in self.device_plugins.items()
            }
        except Exception as e:
            logger.error(f"加载设备插件失败: {e}")
            result["device_plugins"]["error"] = str(e)

        # 加载接口插件
        try:
            self.interface_loader.model_manager = model_manager
            self.interface_plugins = self.interface_loader.load_plugins(model_manager=model_manager)
            result["interface_plugins"] = {
                name: {
                    "status": "loaded",
                    "type": type(plugin).__name__
                }
                for name, plugin in self.interface_plugins.items()
            }
        except Exception as e:
            logger.error(f"加载接口插件失败: {e}")
            result["interface_plugins"]["error"] = str(e)

        self.last_reload_time = time.time()
        return result

    def reload_plugins(self, model_manager=None) -> Dict[str, Dict[str, Any]]:
        """重新加载所有插件"""
        logger.info("重新加载所有插件...")
        self.device_plugins.clear()
        self.interface_plugins.clear()
        return self.load_all_plugins(model_manager)

    def get_device_plugin(self, device_name: str) -> Optional[Any]:
        """获取设备插件"""
        return self.device_plugins.get(device_name)

    def get_interface_plugin(self, interface_name: str) -> Optional[Any]:
        """获取接口插件"""
        return self.interface_plugins.get(interface_name)

    def get_all_device_plugins(self) -> Dict[str, Any]:
        """获取所有设备插件"""
        return self.device_plugins.copy()

    def get_all_interface_plugins(self) -> Dict[str, Any]:
        """获取所有接口插件"""
        return self.interface_plugins.copy()

    def get_plugin_status(self) -> Dict[str, Any]:
        """
        获取所有插件状态
        【修改】现在使用缓存数据来报告设备状态，避免阻塞
        """
        device_status = self.get_device_status_snapshot()
        
        return {
            "device_plugins": {
                name: {
                    "online": data.get("online", False),
                    "type": data.get("type", "Unknown"),
                    "device_info": data.get("info", None)
                }
                for name, data in device_status.items()
            },
            "interface_plugins": {
                name: {
                    "type": type(plugin).__name__,
                    "supported_modes": [name]
                }
                for name, plugin in self.interface_plugins.items()
            },
            "last_reload": self.last_reload_time
        }

    def discover_new_plugins(self) -> Dict[str, List[str]]:
        """发现新插件（不加载）"""
        new_plugins = {
            "device_plugins": [],
            "interface_plugins": []
        }

        # 发现设备插件
        if os.path.exists(self.device_dir):
            for filename in os.listdir(self.device_dir):
                if filename.endswith('.py') and not filename.startswith('__') and filename != 'Base_Class.py':
                    plugin_name = filename[:-3]
                    if plugin_name not in self.device_plugins:
                        new_plugins["device_plugins"].append(plugin_name)

        # 发现接口插件
        if os.path.exists(self.interface_dir):
            for filename in os.listdir(self.interface_dir):
                if filename.endswith('.py') and not filename.startswith('__') and filename != 'Base_Class.py':
                    plugin_name = filename[:-3]
                    if plugin_name not in self.interface_plugins:
                        new_plugins["interface_plugins"].append(plugin_name)

        return new_plugins

    def validate_plugin_structure(self, plugin_path: str) -> Dict[str, Any]:
        """验证插件文件结构"""
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "plugin_class": None
        }

        if not os.path.exists(plugin_path):
            validation_result["errors"].append(f"插件文件不存在: {plugin_path}")
            return validation_result

        try:
            # 尝试加载和验证插件
            spec = importlib.util.spec_from_file_location("temp_plugin", plugin_path)
            if not spec or not spec.loader:
                validation_result["errors"].append("无法创建模块规范")
                return validation_result

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 查找插件类
            found_classes = []

            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, ABC) and
                    obj != ABC and
                    hasattr(obj, '__abstractmethods__')):

                    found_classes.append((name, obj))

            if not found_classes:
                validation_result["errors"].append("未找到有效的插件类")
                return validation_result

            # 选择第一个找到的插件类
            class_name, plugin_class = found_classes[0]
            validation_result["plugin_class"] = class_name

            # 验证抽象方法
            if hasattr(plugin_class, '__abstractmethods__'):
                abstract_methods = plugin_class.__abstractmethods__
                if abstract_methods:
                    validation_result["errors"].append(f"插件类 {class_name} 未实现的抽象方法: {list(abstract_methods)}")
                    return validation_result

            validation_result["valid"] = True
            validation_result["warnings"].append(f"发现 {len(found_classes)} 个潜在插件类，将使用第一个")

        except Exception as e:
            validation_result["errors"].append(f"验证插件失败: {str(e)}")

        return validation_result

    # 便捷方法：直接提供加载器访问
    def get_device_loader(self) -> DevicePluginLoader:
        """获取设备插件加载器"""
        return self.device_loader

    def get_interface_loader(self) -> InterfacePluginLoader:
        """获取接口插件加载器"""
        return self.interface_loader
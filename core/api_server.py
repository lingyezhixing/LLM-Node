from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import asyncio
from typing import Optional
from utils.logger import get_logger
from core.config_manager import ConfigManager
from core.model_controller import ModelController
from core.api_router import APIRouter

logger = get_logger(__name__)

class APIServer:
    """API服务器 - 节点版"""

    def __init__(self, config_manager: ConfigManager, model_controller: ModelController):
        self.config_manager = config_manager
        self.model_controller = model_controller
        self.api_router = APIRouter(self.config_manager, self.model_controller)
        self.app = FastAPI(title="LLM-Manager Node", version="1.0.0")
        self._setup_routes()
        logger.info("API 服务器初始化完成 (节点模式)")

    def _setup_routes(self):
        
        # --- 基础系统接口 ---

        @self.app.get("/api/health")
        async def health_check():
            """节点健康检查，返回运行中的模型数量"""
            # 统计状态为 'routing' 的模型数量
            routing_count = len([s for s in self.model_controller.models_state.values() if s['status'] == 'routing'])
            return {"status": "healthy", "role": "node", "running_models": routing_count}

        @self.app.get("/api/devices/info")
        async def get_device_info():
            """获取节点硬件资源信息 (显存、温度等)"""
            try:
                # 直接从插件系统的缓存中获取，非阻塞
                devices_info = self.model_controller.plugin_manager.get_device_status_snapshot()
                return {"success": True, "devices": devices_info}
            except Exception as e:
                logger.error(f"获取设备信息失败: {e}")
                return {"success": False, "message": str(e)}

        @self.app.get("/v1/models")
        async def list_models():
            """列出节点支持的所有模型配置"""
            return self.model_controller.get_model_list()

        # --- 模型控制接口 (Start/Stop) ---

        @self.app.post("/api/models/{model_alias}/start")
        async def start_model_api(model_alias: str):
            """手动启动指定模型"""
            try:
                model_name = self.config_manager.resolve_primary_name(model_alias)
                # 调用控制器启动模型 (这也是路由自动启动调用的底层方法)
                success, message = self.model_controller.start_model(model_name)
                return {"success": success, "message": message}
            except Exception as e:
                return {"success": False, "message": str(e)}

        @self.app.post("/api/models/{model_alias}/stop")
        async def stop_model_api(model_alias: str):
            """手动停止指定模型"""
            try:
                model_name = self.config_manager.resolve_primary_name(model_alias)
                success, message = self.model_controller.stop_model(model_name)
                return {"success": success, "message": message}
            except Exception as e:
                return {"success": False, "message": str(e)}

        @self.app.post("/api/models/stop-all")
        async def stop_all_models():
            """一键停止所有运行中的模型"""
            try:
                self.model_controller.unload_all_models()
                return {"success": True, "message": "所有模型已关闭"}
            except Exception as e:
                return {"success": False, "message": str(e)}

        # --- 日志流式接口 (Real-time Log Stream) ---

        @self.app.get("/api/models/{model_alias}/logs/stream")
        async def stream_model_logs(model_alias: str):
            """
            实时获取模型日志流 (Server-Sent Events 风格的文本流)
            用于前端或中心节点实时监控模型启动过程和运行输出
            """
            try:
                model_name = self.config_manager.resolve_primary_name(model_alias)
                
                # 1. 验证模型是否存在
                if not self.config_manager.get_model_config(model_name):
                    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

                # 2. 创建异步队列，用于在 EventLoop 中接收来自线程的回调数据
                queue = asyncio.Queue()
                loop = asyncio.get_running_loop()

                # 3. 定义回调函数：ProcessManager (线程) -> LogManager -> Callback -> Queue (EventLoop)
                def log_callback(message: str):
                    # 使用 call_soon_threadsafe 确保跨线程安全地向 async queue 写入
                    loop.call_soon_threadsafe(queue.put_nowait, message)

                # 4. 订阅日志
                self.model_controller.log_manager.subscribe(model_name, log_callback)

                # 5. 定义生成器，持续从队列读取并 yield 给客户端
                async def log_generator():
                    try:
                        while True:
                            # 等待新日志 (await 会释放 CPU 给其他协程)
                            message = await queue.get()
                            yield message
                    except asyncio.CancelledError:
                        # 客户端断开连接 (例如浏览器关闭页面)
                        logger.debug(f"日志流连接断开: {model_name}")
                        pass
                    finally:
                        # 6. 清理订阅，防止内存泄漏
                        self.model_controller.log_manager.unsubscribe(model_name, log_callback)

                return StreamingResponse(log_generator(), media_type="text/plain")

            except Exception as e:
                logger.error(f"建立日志流失败: {e}")
                if isinstance(e, HTTPException):
                    raise e
                raise HTTPException(status_code=500, detail=str(e))

        # --- 核心转发路由 (Catch-All) ---
        # 必须放在最后，捕获所有未匹配的路径，视为对模型的调用 (Chat/Completion/Embedding)
        @self.app.api_route("/{path:path}", methods=["POST", "GET", "PUT", "DELETE", "OPTIONS"])
        async def handle_api_requests(request: Request, path: str):
            return await self.api_router.route_request(request, path)

    def run(self, host: Optional[str] = None, port: Optional[int] = None):
        if host is None or port is None:
            server_cfg = self.config_manager.get_openai_config()
            host = host or server_cfg['host']
            port = port or server_cfg['port']
        logger.info(f"节点接口将在 http://{host}:{port} 上启动")
        uvicorn.run(self.app, host=host, port=port, log_level="warning")

def run_api_server(config_manager: ConfigManager, model_controller: ModelController):
    server = APIServer(config_manager, model_controller)
    server.run()
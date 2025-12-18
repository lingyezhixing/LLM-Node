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
        
        @self.app.get("/api/health")
        async def health_check():
            """节点健康检查，返回运行中的模型数量"""
            routing_count = len([s for s in self.model_controller.models_state.values() if s['status'] == 'routing'])
            return {"status": "healthy", "role": "node", "running_models": routing_count}

        @self.app.get("/api/devices/info")
        async def get_device_info():
            """获取节点硬件资源信息"""
            try:
                devices_info = self.model_controller.plugin_manager.get_device_status_snapshot()
                return {"success": True, "devices": devices_info}
            except Exception as e:
                logger.error(f"获取设备信息失败: {e}")
                return {"success": False, "message": str(e)}

        @self.app.get("/v1/models")
        async def list_models():
            """列出节点支持的模型 (OpenAI 格式)"""
            return self.model_controller.get_model_list()

        # --- 模型查询与控制接口 ---

        @self.app.get("/api/models/{model_alias}/info")
        async def get_model_details(model_alias: str):
            """
            获取模型详细运行信息
            已对齐 LLM-Manager 格式，包含 pending_requests
            """
            try:
                model_name = self.config_manager.resolve_primary_name(model_alias)
                
                # 获取状态和配置
                state = self.model_controller.models_state.get(model_name, {})
                config = self.config_manager.get_model_config(model_name)
                
                if not config:
                     raise HTTPException(status_code=404, detail=f"Model '{model_alias}' not found")

                # 获取待处理请求数 (从 Router 获取)
                pending_requests = self.api_router.pending_requests.get(model_name, 0)

                # 获取进程详细信息 (如果进程存在)
                process_data = None
                if state.get('pid'):
                    process_name = f"model_{model_name}"
                    process_data = self.model_controller.process_manager.get_process_info(process_name)

                # 构造符合 Manager 预期的 model_info 对象
                # Manager 使用: model_status = {**state, "pending_requests": ...}
                model_standard_info = {
                    "status": state.get("status", "unknown"),
                    "pid": state.get("pid"),
                    "last_access": state.get("last_access"),
                    "failure_reason": state.get("failure_reason"),
                    "mode": config.get("mode", "Chat"),
                    "pending_requests": pending_requests,  # 关键字段
                    "port": config.get("port"),
                    "aliases": config.get("aliases", [model_name])
                }

                return {
                    "success": True,
                    # 提供 "model" 键，与 Manager 格式完全对齐
                    "model": model_standard_info,
                    
                    # 保留 Node 版特有的详细调试信息
                    "node_debug_info": {
                        "model_name": model_name,
                        "queried_alias": model_alias,
                        "active_hardware_config": state.get("current_config"),
                        "process_info": process_data
                    }
                }
            except Exception as e:
                logger.error(f"获取模型信息失败: {e}")
                if isinstance(e, HTTPException): raise e
                return {"success": False, "message": str(e)}

        @self.app.post("/api/models/{model_alias}/start")
        async def start_model_api(model_alias: str):
            try:
                model_name = self.config_manager.resolve_primary_name(model_alias)
                success, message = self.model_controller.start_model(model_name)
                return {"success": success, "message": message}
            except Exception as e:
                return {"success": False, "message": str(e)}

        @self.app.post("/api/models/{model_alias}/stop")
        async def stop_model_api(model_alias: str):
            try:
                model_name = self.config_manager.resolve_primary_name(model_alias)
                success, message = self.model_controller.stop_model(model_name)
                return {"success": success, "message": message}
            except Exception as e:
                return {"success": False, "message": str(e)}

        @self.app.post("/api/models/stop-all")
        async def stop_all_models():
            try:
                self.model_controller.unload_all_models()
                return {"success": True, "message": "所有模型已关闭"}
            except Exception as e:
                return {"success": False, "message": str(e)}

        # --- 日志流式接口 ---
        @self.app.get("/api/models/{model_alias}/logs/stream")
        async def stream_model_logs(model_alias: str):
            """
            实时获取模型日志流 (Server-Sent Events 风格的文本流)
            """
            try:
                model_name = self.config_manager.resolve_primary_name(model_alias)
                if not self.config_manager.get_model_config(model_name):
                    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

                queue = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def log_callback(message: str):
                    loop.call_soon_threadsafe(queue.put_nowait, message)

                self.model_controller.log_manager.subscribe(model_name, log_callback)

                async def log_generator():
                    try:
                        while True:
                            message = await queue.get()
                            yield message
                    except asyncio.CancelledError:
                        pass
                    finally:
                        self.model_controller.log_manager.unsubscribe(model_name, log_callback)

                return StreamingResponse(log_generator(), media_type="text/plain")

            except Exception as e:
                logger.error(f"建立日志流失败: {e}")
                if isinstance(e, HTTPException):
                    raise e
                raise HTTPException(status_code=500, detail=str(e))

        # --- 核心转发路由 ---
        # 捕获所有其他请求并转发给本地模型进程
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
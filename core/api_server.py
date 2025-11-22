from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
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
            """列出节点支持的模型"""
            return self.model_controller.get_model_list()

        # --- 模型控制接口 ---

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
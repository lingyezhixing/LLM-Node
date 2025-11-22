from fastapi import Request, HTTPException, Response
from fastapi.responses import StreamingResponse
import json
import httpx
import asyncio
from utils.logger import get_logger
from core.model_controller import ModelController
from core.config_manager import ConfigManager

logger = get_logger(__name__)

class APIRouter:
    """API路由器 - 负责请求路由和转发 (无状态版 - 支持并发冷启动)"""

    def __init__(self, config_manager: ConfigManager, model_controller: ModelController):
        self.config_manager = config_manager
        self.model_controller = model_controller
        self.async_clients = {}
        # 长连接配置，适应大模型推理时间
        self.timeouts = httpx.Timeout(30.0, read=600.0, connect=30.0, write=30.0)

    async def get_async_client(self, port: int):
        """获取复用的异步HTTP客户端"""
        if port not in self.async_clients:
            self.async_clients[port] = httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                timeout=self.timeouts
            )
        return self.async_clients[port]

    async def route_request(self, request: Request, path: str) -> Response:
        """路由请求到目标模型"""
        if request.method == "OPTIONS":
            return Response(status_code=204, headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*"
            })

        # 1. 解析请求体，获取目标模型
        request_data = b''
        model_alias = None
        try:
            # 读取 body
            request_data = await request.body()
            
            # 尝试从 JSON 中提取 model 字段
            if "application/json" in request.headers.get("content-type", "") and request_data:
                body = json.loads(request_data)
                model_alias = body.get("model")
        except Exception as e:
            logger.debug(f"解析请求体失败: {e}")

        if not model_alias:
            raise HTTPException(status_code=400, detail="Request body must contain 'model' field")

        # 2. 解析模型配置
        try:
            model_name = self.config_manager.resolve_primary_name(model_alias)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Model alias '{model_alias}' not found")

        model_config = self.config_manager.get_model_config(model_name)
        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' config not found")

        # 3. 验证接口插件
        model_mode = model_config.get("mode", "Chat")
        interface_plugin = self.model_controller.plugin_manager.get_interface_plugin(model_mode)
        if not interface_plugin:
            raise HTTPException(status_code=400, detail=f"Unsupported model mode: {model_mode}")

        is_valid, error_message = interface_plugin.validate_request(path, model_name)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)

        # 4. 确保模型已启动 (节点核心功能：按需启动)
        # 【优化】使用 asyncio.to_thread 确保启动过程不阻塞 API 主循环
        # 这使得多个请求可以触发多个模型的并行冷启动
        try:
            success, message = await asyncio.to_thread(self.model_controller.start_model, model_name)
            if not success:
                raise HTTPException(status_code=503, detail=message)

            # 5. 转发请求
            target_port = model_config['port']
            client = await self.get_async_client(target_port)
            target_url = client.base_url.join(path)

            # 清理 headers
            headers = dict(request.headers)
            headers.pop("host", None)
            headers.pop("content-length", None)
            headers.pop("transfer-encoding", None)

            req = client.build_request(
                request.method,
                target_url,
                headers=headers,
                content=request_data,
                params=request.query_params
            )

            response = await client.send(req, stream=True)
            
            # 直接流式透传响应，不做任何处理
            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers=dict(response.headers)
            )

        except Exception as e:
            logger.error(f"Error handling request for '{model_name}': {e}", exc_info=True)
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Internal Node Error: {str(e)}")
from fastapi import Request, HTTPException, Response
from fastapi.responses import StreamingResponse
import json
import httpx
import asyncio
import time
from typing import Set  # 添加 Set 类型导入
from utils.logger import get_logger
from core.model_controller import ModelController
from core.config_manager import ConfigManager

logger = get_logger(__name__)

class APIRouter:
    """API路由器 - 负责请求路由和转发 (无状态版 - 已修复并发死锁)"""

    def __init__(self, config_manager: ConfigManager, model_controller: ModelController):
        self.config_manager = config_manager
        self.model_controller = model_controller
        self.async_clients = {}
        # 长连接配置，适应大模型推理时间
        self.timeouts = httpx.Timeout(30.0, read=600.0, connect=30.0, write=30.0)
        
        # 【新增】本地启动任务标记，防止高并发请求耗尽线程池
        self.starting_models: Set[str] = set()

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
        # 【核心修复】引入异步等待循环，防止高并发启动耗尽线程池
        try:
            # 定义启动过程中的过渡状态
            STARTUP_STATES = ['starting', 'init_script', 'health_check']

            while True:
                # A. 获取当前模型状态 (原子操作)
                model_state = self.model_controller.models_state.get(model_name, {})
                current_status = model_state.get('status', 'stopped')

                # B. 如果模型已运行，跳出循环进行转发
                if current_status == 'routing':
                    break

                # C. 检查是否正在启动
                # 全局状态检查
                is_starting_global = current_status in STARTUP_STATES
                # 本地路由锁检查 (闭合并发时间窗口)
                is_starting_local = model_name in self.starting_models

                if is_starting_global or is_starting_local:
                    # 发现正在启动，异步休眠等待，让出 CPU 给 Event Loop
                    # 这确保了 100 个请求只会占用 0 个线程在等待
                    logger.debug(f"[NODE_ROUTER] 模型 {model_name} 正在启动中({current_status})，异步等待...")
                    await asyncio.sleep(0.5)
                    continue

                # D. 只有状态为停止/失败，且本地没有正在进行的启动任务时，才发起启动
                if current_status in ['stopped', 'failed']:
                    # 标记本地锁
                    self.starting_models.add(model_name)
                    try:
                        logger.info(f"[NODE_ROUTER] 模型 {model_name} 需要启动，分配唯一启动线程...")
                        # 这是一个耗时操作，占用 1 个线程
                        success, message = await asyncio.to_thread(
                            self.model_controller.start_model, model_name
                        )
                        if not success:
                            raise HTTPException(status_code=503, detail=message)
                        # 启动成功后，下一次循环会检测到 routing 状态并 break
                    except Exception as e:
                        logger.error(f"[NODE_ROUTER] 启动模型异常: {e}")
                        if isinstance(e, HTTPException):
                            raise e
                        raise HTTPException(status_code=503, detail=f"启动异常: {str(e)}")
                    finally:
                        # 无论成功失败，移除本地标记
                        if model_name in self.starting_models:
                            self.starting_models.remove(model_name)
                    continue
                
                # 其他未知状态兜底等待
                await asyncio.sleep(0.5)

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
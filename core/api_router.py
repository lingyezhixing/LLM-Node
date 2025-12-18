from fastapi import Request, HTTPException, Response
from fastapi.responses import StreamingResponse
import json
import httpx
import asyncio
import time
from typing import Set, Dict
from utils.logger import get_logger
from core.model_controller import ModelController
from core.config_manager import ConfigManager

logger = get_logger(__name__)

class APIRouter:
    """API路由器 - 负责请求路由和转发 (节点版 - 已修复请求计数与空闲检测)"""

    def __init__(self, config_manager: ConfigManager, model_controller: ModelController):
        self.config_manager = config_manager
        self.model_controller = model_controller
        self.async_clients = {}
        # 长连接配置，适应大模型推理时间
        self.timeouts = httpx.Timeout(30.0, read=600.0, connect=30.0, write=30.0)
        
        # 本地启动任务标记，防止高并发请求耗尽线程池
        self.starting_models: Set[str] = set()
        
        # 请求计数器 (用于负载均衡和空闲检测)
        self.pending_requests: Dict[str, int] = {}

    def _touch_model_activity(self, model_name: str):
        """更新模型的最后活动时间戳，防止在处理请求时被误判为空闲"""
        if model_name in self.model_controller.models_state:
            state = self.model_controller.models_state[model_name]
            # 获取锁更新时间戳，确保线程安全
            with state['lock']:
                state['last_access'] = time.time()

    def increment_pending_requests(self, model_name: str):
        """增加待处理请求计数"""
        if model_name not in self.pending_requests:
            self.pending_requests[model_name] = 0
        self.pending_requests[model_name] += 1
        
        # 请求到达时更新活动时间
        self._touch_model_activity(model_name)
        
        logger.info(f"[NODE_ROUTER] 模型 {model_name} 新请求进入，当前待处理: {self.pending_requests[model_name]}")

    def mark_request_completed(self, model_name: str):
        """标记请求完成"""
        if model_name in self.pending_requests:
            self.pending_requests[model_name] = max(0, self.pending_requests[model_name] - 1)
            
            # 请求结束时再次更新活动时间 (倒计时重置)
            self._touch_model_activity(model_name)
            
            logger.info(f"[NODE_ROUTER] 模型 {model_name} 请求完成，剩余待处理: {self.pending_requests[model_name]}")

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

        # 4. 增加并发计数 (必须在耗时操作前)
        self.increment_pending_requests(model_name)

        try:
            # 5. 确保模型已启动 (节点核心功能：按需启动)
            # 引入异步等待循环，防止高并发启动耗尽线程池
            
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

            # 6. 转发请求
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
            
            # 使用 Wrapper 包装流式响应，以正确减少计数
            async def stream_wrapper():
                try:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                except Exception as e:
                    logger.error(f"[NODE_ROUTER] 流传输异常: {e}")
                    raise e
                finally:
                    # 无论成功还是异常断开，都在这里减少计数
                    await response.aclose()
                    self.mark_request_completed(model_name)

            return StreamingResponse(
                stream_wrapper(),
                status_code=response.status_code,
                headers=dict(response.headers)
            )

        except Exception as e:
            # 发生异常时也要减少计数，防止死锁
            self.mark_request_completed(model_name)
            logger.error(f"Error handling request for '{model_name}': {e}", exc_info=True)
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Internal Node Error: {str(e)}")
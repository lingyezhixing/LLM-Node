# LLM-Node

**LLM-Node** 是一个轻量级、无状态的本地 LLM 计算节点服务。

本项目是从 **LLM-Manager** 项目中剥离出的独立分支，并在 2026-08 依据完全重构后的
LLM-Manager v3 架构重写。它保留了核心的模型进程管理、硬件资源调度和统一 API 路由，
移除了 WebUI、SQLite 数据库、计费/用量模块和系统托盘，作为纯粹的后端计算节点运行。

> **⚠️ 说明**：
> 本项目为个人开发工具，主要用于构建无头（Headless）推理环境。
> 仅提供模型托管和接口转发功能，不包含用户界面。

---

## 核心特性

1. **纯后端运行**：无 GUI、无系统托盘，专为服务器和容器环境设计。
2. **无状态架构**：无数据库依赖，配置走 YAML，启动即用，无历史负担。
3. **统一接口**：提供兼容 OpenAI 格式的 API 入口，自动路由至后端模型端口。
4. **按需调度**：请求触发启动 + 空闲自动关闭 + 显存不足时动态驱逐（对齐 Manager v3 逻辑）。
5. **并发安全**：单派发 Future 去重 + 全局 spawn 锁 + owner-token guard。
6. **健康探测**：按模式（Chat / Embedding / Reranker）两阶段探测（浅层 /v1/models + 深层请求）。
7. **设备监控**：nvidia-smi（NVIDIA）、amdgpu sysfs / LHM（AMD）、i915 + intel_gpu_top / LHM（Intel）、psutil（CPU），模糊匹配设备名。
8. **结构化启动**：YAML 中每个 scheme 定义 `script_path`（.bat/.sh），支持 `{{port}}` / `{{alias}}` 变量替换。
9. **容器化支持**：原生支持 Docker 和 Docker Compose 部署。

---

## 架构

```
config     ── YAML → frozen dataclasses + 校验(无 DB;config_store 读穿快照)
state      ── 内存状态机 + 单派发 inflight Future + activity
supervisor ── 子进程管理(kill_tree + 单 wait 协程 + 输出泵线程)
runtime    ── lifecycle(启动/停止 pipeline + 协作式中断)/ scheduling(纯函数资源决策)/ background(空闲回收 + 自动启动)
devices    ── DeviceMonitor + 4 平台适配器(NVIDIA/AMD/Intel/CPU)
gateway    ── OpenAI 兼容代理 catch-all + 管理 API + 别名解析
model_log  ── 模型输出文件日志(无 DB,每模型保留最新 10 个文件)
```

- **单进程模型**：一个 Python 进程跑一个 app（FastAPI + uvicorn），模块级单例内存状态。
- **配置单一源**：`config.yaml`，运行时只读 frozen 快照；编辑后 `ConfigStore.reload()` 即时生效（读穿）。
- **无状态**：不落任何数据库，无计费，无 WebUI。模型 stdout/stderr 仅落文件日志。

---

## 快速开始

### 1. 环境准备
*   Python 3.11+
*   或 Docker 环境

### 2. 安装
```bash
# 仅运行(NVIDIA nvidia-smi 默认可用;AMD/Intel 核显监控需 monitoring)
pip install -e ".[monitoring]"
# 开发 / 测试
pip install -e ".[monitoring,dev]"
```

### 3. 配置文件 (`config.yaml`)
仓库只提供样板 `config.yaml.example`,实际配置文件由本地复制生成,按本机硬件/模型
路径修改,可随意调整而不影响 git:
```bash
cp config.yaml.example config.yaml
# 编辑 config.yaml(模型/端口/脚本路径),然后运行
```
关键字段:
- `aliases[0]` = 下游 served name(客户端调用名)
- `mode` ∈ `Chat` / `Embedding` / `Reranker`(决定健康探测方式)
- `script_path` 支持 `{{port}}` / `{{alias}}` 变量替换;Windows 填 `.bat`,Linux 填 `.sh`
- `memory_mb` 用于显存 deficit 计算与驱逐决策,需如实填写
- 设备不满足前一个 scheme 时自动回退到下一个(多 GPU 启动灵活性)

### 4. 运行方式

#### 方式 A: 直接运行 (Python)
```bash
python main.py
# 或:python -m llm_node
```

#### 方式 B: Docker Compose
仓库只提供样板(`Dockerfile.example` / `docker-compose.yml.example`),实际部署文件
由本地复制生成,可随意修改而不影响 git:
```bash
cp Dockerfile.example Dockerfile
cp docker-compose.yml.example docker-compose.yml
# 按需编辑 docker-compose.yml(端口/设备/外部网络等),然后:
docker-compose up -d
```

---

## API 接口说明

*   **业务接口**:
    *   `/v1/chat/completions`: 对话补全 (自动路由)
    *   `/v1/embeddings`: 向量嵌入 (自动路由)
    *   `/v1/rerank`: 重排序 (自动路由)
    *   `/v1/models`: 列出可用模型

*   **管理接口**:
    *   `POST /api/models/{alias}/start`: 预热/启动模型
    *   `POST /api/models/{alias}/stop`: 停止模型
    *   `POST /api/models/{alias}/restart`: 重启模型
    *   `POST /api/models/stop-all`: 停止所有模型
    *   `GET /api/models`: 列出模型及运行状态
    *   `GET /api/models/{alias}/info`: 获取模型运行状态
    *   `GET /api/health`: 节点健康检查

---

## 开发

后端（项目根）：
```bash
python -m pytest tests -q     # 全量测试
ruff format --check .         # 格式
ruff check .                  # lint
python -m pyright src/llm_node  # 类型检查
```

---

## 更新日志 (Changelog)

### v2.0.0 - 2026-08-07
**依据重构后的 LLM-Manager v3 架构重写**
*   **架构对齐**：采用与 Manager v3 相同的分层结构（config/state/supervisor/runtime/devices/gateway），
    保留无状态定位（无 DB、无计费、无 WebUI、无托盘）。
*   **生命周期**：移植单派发 + 全局 spawn 锁 + owner-token guard + 协作式中断 + 崩溃恢复 + reconcile 兜底。
*   **设备监控**：移植 DeviceMonitor + 四平台适配器（nvidia-smi / amdgpu+LHM / i915+intel_gpu_top / psutil）。
*   **调度**：移植纯函数 scheduling（显存 deficit 计算 + 空闲/显存加权驱逐）。
*   **健康探测**：移植两阶段 probe（浅层 /v1/models + 深层按模式请求）。
*   **代理转发**：移植 OpenAI 兼容 catch-all 代理，剥离 token 计量。
*   **模型日志**：模型 stdout/stderr 落文件日志（`logs/model_logs/<model>/`，每模型保留 10 个文件），替代原 DB 日志。
*   **测试**：移植核心单测（config/state/supervisor/probes/devices/scheduling/background/lifecycle/proxy/routes/api）+ smoke，226 项。

# LLM-Node

**LLM-Node** 是一个轻量级、无状态的本地 LLM 计算节点服务。它是从 **LLM-Manager** 项目剥离出的独立分支，并按重构后的 LLM-Manager v3 架构重写：保留核心的模型进程管理、硬件资源调度和统一 API 网关，移除 WebUI、SQLite 数据库、计费/用量模块和系统托盘，作为纯后端计算节点运行。

> **⚠️ 说明**：本项目为个人开发工具，主要用于构建无头（Headless）推理环境，仅提供模型托管和接口转发，不包含用户界面。

---

## 特性

1. **纯后端运行**：无 GUI、无系统托盘，面向服务器与容器环境。
2. **无状态架构**：不落任何数据库、无计费，配置走 YAML，启动即用，无历史负担。
3. **统一 API 网关**：透明代理，支持 OpenAI / Anthropic Claude / Responses 三种兼容格式，按 `model` 字段自动路由至对应模型端口。
4. **按需调度**：请求触发冷启动、空闲自动回收、显存不足时动态驱逐（对齐 Manager v3 逻辑）。
5. **并发安全**：单派发 Future 去重 + 全局 spawn 锁 + owner-token guard，高并发/并发重启不乱序、不重复拉起。
6. **健康探测**：按模型模式（Chat / Embedding / Reranker）两阶段探测（浅层 `/v1/models` + 深层按模式请求）。
7. **设备监控**：NVIDIA（nvidia-smi）、AMD（amdgpu sysfs / LHM）、Intel（i915 + intel_gpu_top / LHM）、CPU（psutil）四平台适配器，模糊匹配设备名。
8. **脚本化启动**：YAML 中每个 scheme 定义 `script_path`（`.bat` / `.sh`），支持 `{{port}}` / `{{alias}}` 变量替换，多 scheme 按设备回退。
9. **容器友好**：提供 Docker / Compose 样板，镜像内不打包模型与配置。

---

## 架构

```
config     ── YAML → frozen dataclasses + 校验（config_store 读穿快照）
state      ── 内存状态机 + 单派发 inflight Future + activity
supervisor ── 子进程管理（kill_tree + 单 wait 协程 + 输出泵线程）
runtime    ── lifecycle（启动/停止 pipeline + 协作式中断）· scheduling（纯函数资源决策）· background（空闲回收 + 自动启动）
devices    ── DeviceMonitor + 四平台适配器（NVIDIA/AMD/Intel/CPU）
gateway    ── OpenAI/Anthropic 兼容代理 catch-all + 管理 API + 别名解析
model_log  ── 模型输出文件日志（无 DB，每模型保留最新 10 个文件）
```

- **单进程模型**：一个 Python 进程运行一个 app（FastAPI + uvicorn），模块级单例内存状态，无额外守护进程。
- **配置单一源**：仓库只跟踪样板 `config.yaml.example`，实际 `config.yaml` 由本地复制生成，各机可自由维护；运行时只读 frozen 快照，编辑后 `ConfigStore.reload()` 即时生效（读穿）。
- **无状态**：模型 stdout/stderr 仅落文件日志，停止即清空内存状态，重启无历史负担。

---

## 快速开始

### 环境要求

- Python 3.11+
- 或 Docker 环境

### 安装

```bash
# 仅运行（NVIDIA 的 nvidia-smi 默认可用；AMD/Intel 核显监控需 monitoring）
pip install -e ".[monitoring]"

# 开发 / 测试
pip install -e ".[monitoring,dev]"
```

### 配置

仓库只提供样板 `config.yaml.example`，实际配置文件由本地复制生成，按本机硬件/模型路径修改，可随意调整而不影响 git：

```bash
cp config.yaml.example config.yaml
```

`config.yaml` 结构要点：

- **program**：监听地址/端口、`alive_time`（模型空闲自动关闭分钟数，`<=0` 禁用）、日志级别。
- **Local-Models** 下每个模型定义：
  - `aliases[0]`：下游 served name，即客户端调用时的 `model` 字段值；
  - `mode`：`Chat` / `Embedding` / `Reranker`，决定健康探测方式；
  - `script_path`：启动脚本路径（Windows 填 `.bat`，Linux 填 `.sh`），支持 `{{port}}` / `{{alias}}` 变量替换；
  - `memory_mb`：按设备声明显存占用（MB），用于显存 deficit 计算与驱逐决策，需如实填写；
  - 多个 scheme 按配置顺序回退——当设备的 `required_devices` 不满足当前 scheme 时自动尝试下一个（多 GPU 启动灵活性）。

### 运行

**方式 A：直接运行（Python）**

```bash
python main.py            # 或 python -m llm_node
# 默认读取 config.yaml；可用环境变量覆盖：LLM_NODE_CONFIG=/path/to/config.yaml
```

**方式 B：Docker Compose**

仓库同样只提供样板 `Dockerfile.example` / `docker-compose.yml.example`，实际部署文件由本地复制生成：

```bash
cp Dockerfile.example Dockerfile
cp docker-compose.yml.example docker-compose.yml
# 按需编辑 docker-compose.yml（端口 / 设备 / 外部网络等），然后：
docker-compose up -d
```

---

## API 接口

### 业务接口（透明代理网关）

节点对 `POST / PUT / DELETE / PATCH /{path:path}` 的任意请求都会：解析 JSON 中的 `model` 字段 → 按别名路由到对应模型端口 → 原样转发（并把 `model` 自动重写为下游 served name）。因此与主项目一致支持三种兼容格式：

- **OpenAI**：`/v1/chat/completions`、`/v1/completions`
- **Embedding / Reranker**：`/v1/embeddings`、`/v1/rerank`
- **Anthropic Claude**：`/v1/messages`
- **OpenAI Responses**：`/v1/responses`

> 只要 JSON 请求体含 `model` 字段即可路由；Anthropic / Responses 等格式按原样透传，实际可用性取决于下游模型后端（lmdeploy / llama.cpp 等）是否实现对应端点。

| 路径 | 说明 |
|---|---|
| `GET /v1/models` | 列出可用模型（`id` = `aliases[0]`，节点直接返回，不转发） |

### 管理接口（节点自身，不经代理）

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/models/{alias}/start` | 预热 / 启动模型 |
| `POST` | `/api/models/{alias}/stop` | 停止模型 |
| `POST` | `/api/models/{alias}/restart` | 重启模型 |
| `POST` | `/api/models/stop-all` | 停止所有模型 |
| `GET` | `/api/models` | 列出模型及运行状态 |
| `GET` | `/api/models/{alias}/info` | 获取单个模型运行状态 |
| `GET` | `/health` | 节点健康检查 |

---

## 开发

后端（项目根目录）：

```bash
python -m pytest tests -q        # 全量测试
ruff format --check .            # 格式检查
ruff check .                     # lint
python -m pyright src/llm_node   # 类型检查
```

---

## 目录结构

```
src/llm_node/
├── app.py              # 组合根：create_app / lifespan / run
├── config.py           # YAML → frozen dataclasses + 校验
├── config_store.py     # YAML 读穿快照
├── state.py            # 内存状态机 + 单派发
├── supervisor.py       # 子进程管理
├── probes.py           # 按模式健康探测
├── model_log.py        # 模型输出文件日志
├── logging_setup.py    # 日志配置
├── devices/            # 设备监控适配器（NVIDIA/AMD/Intel/CPU）
├── runtime/            # lifecycle / scheduling / background
└── gateway/            # 代理 catch-all + 管理 API + 别名解析
```

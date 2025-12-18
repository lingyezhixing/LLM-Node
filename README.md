# LLM-Node

**LLM-Node** 是一个轻量级、无状态的本地 LLM 计算节点服务。

本项目是从 **LLM-Manager** 项目中剥离出的独立分支。它保留了核心的模型进程管理、硬件资源调度和统一 API 路由功能，但移除了所有前端界面 (WebUI)、数据库依赖和计费统计模块。旨在作为一个纯粹的后端服务或集群中的计算节点运行，支持 Docker 部署。

> **⚠️ 说明**：
> 本项目为个人开发工具，主要用于构建无头（Headless）推理环境。
> 仅提供模型托管和接口转发功能，不包含用户界面。

---

## 核心特性

1.  **纯后端运行**：无 GUI、无系统托盘，专为服务器和容器环境设计。
2.  **无状态架构**：移除 SQLite 数据库依赖，仅保留文件日志，启动即用，无历史负担。
3.  **统一接口**：提供兼容 OpenAI 格式的 API 入口，自动路由至后端模型端口。
4.  **按需调度**：保留了完整的进程管理逻辑，支持请求触发启动和空闲自动关闭。
5.  **容器化支持**：原生支持 Docker 和 Docker Compose 部署。
6.  **配置升级**：采用更易读的 YAML 格式进行配置管理。

---

## 快速开始

### 1. 环境准备
*   Python 3.10+
*   或者 Docker 环境

### 2. 配置文件 (`config.yaml`)
LLM-Node 仅需要极简的配置。请复制 `config.example.yaml` 为 `config.yaml`：

```yaml
program:
  host: "0.0.0.0"
  port: 8080
  log_level: "INFO"
  alive_time: 30          # 模型空闲自动关闭时间（分钟）
  # device_plugin_dir: "plugins/devices" # 可选：指定插件目录

Local-Models:
  # 模型配置示例
  Qwen-14B:
    aliases: ["qwen-14b", "gpt-3.5-turbo"]
    mode: "Chat"
    port: 10001
    auto_start: false
    
    # 硬件配置方案
    Standard-Config:
      required_devices: ["rtx 4060"]
      script_path: "scripts/run_qwen.bat" # Linux下请填写 .sh 路径
      memory_mb:
        "rtx 4060": 12000
```

### 3. 运行方式

#### 方式 A: 直接运行 (Python)
```bash
pip install -r requirements.txt
python main.py
```

#### 方式 B: Docker Compose
```bash
docker-compose up -d
```

---

## API 接口说明

LLM-Node 仅保留了最核心的模型控制接口：

*   **业务接口**:
    *   `/v1/chat/completions`: 对话补全 (自动路由)
    *   `/v1/embeddings`: 向量嵌入 (自动路由)
    *   `/v1/rerank`: 重排序 (自动路由)
    *   `/v1/models`: 列出可用模型

*   **管理接口**:
    *   `POST /api/models/{alias}/start`: 预热/启动模型
    *   `POST /api/models/{alias}/stop`: 停止模型
    *   `POST /api/models/stop-all`: 停止所有模型
    *   `GET /api/models/{alias}/info`: 获取模型运行状态
    *   `GET /api/health`: 节点健康检查

---

## 更新日志 (Changelog)

### v1.1.0 - 2025-12-18
**稳定性修复**
*   **[Critical]** 修复了空闲检查时间不更新导致的意外关闭问题。
*   **[Critical]** 升级了关闭模型的逻辑，采用动态加权实现更精细的控制。

### v1.0.3 - 2025-12-03
**稳定性修复**
*   **[Critical]** 修复了高并发请求（如 30+ QPS）触发模型冷启动时，导致线程池耗尽（Thread Pool Starvation）从而引发节点假死的问题。
*   在 Router 层引入本地异步锁，优化了并发启动逻辑。

### v1.0.2 - 2025-11-23 (晚间)
**配置与容器化升级**
*   **配置迁移**：将配置文件从 JSON 全面迁移至 YAML 格式，提升可读性。
*   **Docker 支持**：完善了 `Dockerfile` 和 `docker-compose.yml`，支持一键部署。
*   **接口精简**：移除了与 WebUI 相关的所有端口映射和冗余接口，仅保留核心管理 API。

### v1.0.1 - 2025-11-23 (下午)
**清理与适配**
*   **代码剥离**：移除了 WebUI 构建文件、数据库监控模块和 GPU 插件中不必要的依赖。
*   **依赖修正**：修复了因环境剥离导致的库缺失问题。
*   **兼容性**：修复了 Windows 路径字符导致的编码错误，初步尝试 Linux 路径适配。

### v1.0.0 - 2025-11-23 (初始版本)
**项目独立**
*   **Fork/Split**：从 LLM-Manager 项目（模型管理器重构后版本）正式剥离。
*   **初始化**：确立无状态节点架构，规范化日志记录格式。
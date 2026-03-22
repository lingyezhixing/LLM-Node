# 使用官方 Python 3.12 轻量级镜像
FROM python:3.12-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. [固化] 替换 APT 源为阿里云 (为了速度)
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 2. [最小化安装] 系统依赖
# --no-install-recommends: 不安装推荐包，减小体积
RUN apt-get update && apt-get install -y --no-install-recommends \
    # === Python 运行依赖（原有）===
    libgomp1 \
    libopenblas0 \
    # === Vulkan 运行时（原有，用于 llm-node 推理）===
    mesa-vulkan-drivers \
    libvulkan1 \
    # === Vulkan 编译依赖（新增，用于 llama.cpp 编译）===
    libvulkan-dev \
    glslc \
    # === CMake 构建工具（新增）===
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 4. 授权并启动脚本
RUN echo '#!/bin/bash\n\
if [ -d "Model_startup_script" ]; then\n\
    echo ">>> checking and fixing scripts..."\n\
    find Model_startup_script -name "*.sh" -exec sed -i "s/\r$//" {} \;\n\
    find Model_startup_script -name "*.sh" -exec chmod +x {} \;\n\
fi\n\
\n\
echo ">>> Starting LLM-Node..."\n\
exec "$@"' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "main.py"]
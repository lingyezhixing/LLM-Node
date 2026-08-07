# 使用官方 Python 3.12 轻量级镜像
FROM python:3.12-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. 替换 APT 源为阿里云
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 2. [最小化安装] 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    mesa-vulkan-drivers \
    libvulkan1 \
    libvulkan-dev \
    glslc \
    cmake \
    build-essential \
    ccache \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 安装 Python 依赖(editable install 需要 pyproject + 源码)
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 4. 启动脚本（自动配置 Git 代理 + 权限修复）
RUN echo '#!/bin/bash\n\
echo "[GIT-PROXY] Setting up proxy via v2raya:20171"\n\
git config --global http.proxy "http://v2raya:20171"\n\
git config --global https.proxy "http://v2raya:20171"\n\
if [ -d "Model_startup_script" ]; then\n\
    echo ">>> checking and fixing scripts..."\n\
    find Model_startup_script -name "*.sh" -exec sed -i "s/\r$//" {} \;\n\
    find Model_startup_script -name "*.sh" -exec chmod +x {} \;\n\
fi\n\
echo ">>> Starting LLM-Node..."\n\
exec "$@"' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "main.py"]

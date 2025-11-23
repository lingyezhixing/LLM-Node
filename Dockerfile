# 使用官方 Python 3.12 轻量级镜像
FROM python:3.12-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. [固化] 替换 APT 源为阿里云
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 2. [固化] 安装系统依赖
# - libgomp1: 修复 "no CPU backend found"
# - libopenblas-dev: 基础数学库
# - procps/curl: 调试工具
RUN apt-get update && apt-get install -y \
    procps \
    curl \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 安装 Python 依赖 (使用阿里云 PyPI)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 4. [核心修改] 创建增强版启动脚本
# 包含：
# - sed -i 's/\r$//' : 自动将 Windows 换行符 (CRLF) 转换为 Linux (LF)
# - chmod +x         : 自动赋予执行权限
RUN echo '#!/bin/bash\n\
if [ -d "Model_startup_script" ]; then\n\
    echo ">>> checking and fixing scripts..."\n\
    # 1. 修复换行符 (防止 /bin/sh^M: bad interpreter)\n\
    find Model_startup_script -name "*.sh" -exec sed -i "s/\r$//" {} \;\n\
    # 2. 赋予执行权限\n\
    find Model_startup_script -name "*.sh" -exec chmod +x {} \;\n\
fi\n\
\n\
echo ">>> Starting LLM-Node..."\n\
exec "$@"' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "main.py"]
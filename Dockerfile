# 使用官方 Python 3.12 轻量级镜像
FROM python:3.12-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------
# [加速配置] 替换 APT 源为阿里云镜像 (Debian 12 Bookworm)
# ---------------------------------------------------------
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装基础系统工具
RUN apt-get update && apt-get install -y \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 1. 仅复制依赖文件
COPY requirements.txt .

# ---------------------------------------------------------
# [加速配置] 使用阿里云 PyPI 镜像源安装依赖
# ---------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 3. 创建启动脚本 (Entrypoint) - 处理权限
RUN echo '#!/bin/bash\n\
# 检查脚本目录是否存在\n\
if [ -d "Model_startup_script" ]; then\n\
    echo ">>> Setting permissions for .sh files..."\n\
    find Model_startup_script -name "*.sh" -exec chmod +x {} \;\n\
fi\n\
\n\
# 执行传入的命令\n\
exec "$@"' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

# 设置入口点
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# 默认启动命令
CMD ["python", "main.py"]
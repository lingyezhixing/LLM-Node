# 使用官方 Python 3.12 轻量级镜像
FROM python:3.12-slim

# 设置环境变量
# 防止 Python 生成 .pyc 文件
ENV PYTHONDONTWRITEBYTECODE=1
# 确保日志直接输出到控制台
ENV PYTHONUNBUFFERED=1

# 安装基础系统工具 (psutil 等库可能需要 procps, curl 用于调试)
RUN apt-get update && apt-get install -y \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 1. 仅复制依赖文件 (利用 Docker 缓存层)
COPY requirements.txt .

# 2. 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 3. 创建启动脚本 (Entrypoint)
# 因为代码是挂载进来的，必须在运行时修改权限，而不是构建时
RUN echo '#!/bin/bash\n\
# 检查脚本目录是否存在\n\
if [ -d "Model_startup_script" ]; then\n\
    echo ">>> Setting permissions for .sh files..."\n\
    find Model_startup_script -name "*.sh" -exec chmod +x {} \;\n\
fi\n\
\n\
# 执行传入的命令 (即 CMD)\n\
exec "$@"' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

# 设置入口点
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# 默认启动命令
CMD ["python", "main.py"]
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. [固化] 换源
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 2. [固化] 安装 libgomp1 解决 no CPU backend 错误
RUN apt-get update && apt-get install -y \
    procps \
    curl \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# ... (后面保持不变) ...
RUN echo '#!/bin/bash\n\
if [ -d "Model_startup_script" ]; then\n\
    find Model_startup_script -name "*.sh" -exec chmod +x {} \;\n\
fi\n\
exec "$@"' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "main.py"]
# ---------- Base ----------
FROM python:3.11-slim

# 基础环境
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    TRANSFORMERS_CACHE=/opt/hf \
    TOKENIZERS_PARALLELISM=false \
    PORT=8080

WORKDIR /app

# 如果有需要编译的库，装最小构建依赖（没有就可删掉这一段）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

# 先拷 requirements 以利用缓存
COPY requirements.txt .

# 安装依赖（包含 torch / transformers / sentence-transformers 等）
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 再拷贝项目代码（包含 models/ 等本地模型目录）
COPY . .

# Railway 会注入 $PORT，这里保持兼容
EXPOSE 8080

# 用 shell 形式 CMD 以便变量展开（JSON 数组不会展开 ${PORT}）
CMD gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT:-8080} --workers ${WORKERS:-2} --timeout 120

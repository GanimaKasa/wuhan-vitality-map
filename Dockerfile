FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# 先装CPU-only的torch（部署环境没有GPU，PyPI默认wheel会带CUDA运行时，体积能差好几倍），
# 再装其余依赖，pip发现torch已满足就不会覆盖。
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/

# 构建时把embedding模型一起下载进镜像，容器启动不用再联网拉取，冷启动更快更稳
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5', device='cpu')"

WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

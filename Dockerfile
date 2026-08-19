FROM python:3.11-slim

WORKDIR /app

COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/src/ backend/src/
# -e（可编辑安装）是刻意的：core/config.py靠Path(__file__).resolve().parents[N]
# 推算data/、frontend/的绝对路径，普通pip install会把包复制进site-packages，
# __file__就不再指向/app/backend/src了，这套相对路径推算会全部失效。可编辑安装
# 让__file__继续指向源码原始位置，路径推算才成立。
RUN pip install --no-cache-dir -e ./backend

COPY backend/data/ backend/data/
COPY frontend/ frontend/

WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "vitality_map.main:app", "--host", "0.0.0.0", "--port", "8000"]

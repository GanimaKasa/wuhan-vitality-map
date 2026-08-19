# ==============================================================
#  FastAPI应用入口：只做"实例化app+挂中间件+挂路由+挂静态前端"这一件事，
#  业务逻辑全部在api/、tools/、agents/、retrieval/、services/包里。
#  对应旧的backend/app.py（那一个文件曾经把所有东西都塞在一起）。
# ==============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from vitality_map.api import calendar, chat, geojson, vitality, weather, weibo
from vitality_map.core.config import FRONTEND_DIR
from vitality_map.core.logging import setup_logging

setup_logging()

app = FastAPI(title="武汉城市活力地图")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
# 全量POI接口一次性传约9MB JSON，开GZip压缩能大幅减小实际传输体积，减轻Render免费档带宽压力。
app.add_middleware(GZipMiddleware, minimum_size=1000)

for router_module in (geojson, calendar, weather, vitality, weibo, chat):
    app.include_router(router_module.router)

# 必须放在最后——StaticFiles挂载在"/"，会兜底接管所有没被上面路由匹配到的请求
# （包括前端的index.html/app.js/style.css），挂早了会抢先吃掉/api/*的请求。
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

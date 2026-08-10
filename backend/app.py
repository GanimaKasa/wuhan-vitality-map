# ==============================================================
#  FastAPI 主服务：地图数据 / 查询检索 / LLM问答(mock) / 前端静态文件
# ==============================================================

import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import llm_client  # 导入时会load_dotenv()，读取.env里的DEEPSEEK_API_KEY/HF_TOKEN
import retrieval

# /api/chat 限流：每IP每分钟最多CHAT_RATE_LIMIT次，防止公网调用真实LLM API被刷爆费用。
# 简单内存计数，进程重启会清零；如果部署在反向代理后面，需要改成读X-Forwarded-For。
CHAT_RATE_LIMIT = 10
CHAT_RATE_WINDOW_SECONDS = 60
_chat_request_log = defaultdict(list)  # ip -> 请求时间戳列表


def _check_rate_limit(ip: str):
    now = time.time()
    window_start = now - CHAT_RATE_WINDOW_SECONDS
    timestamps = _chat_request_log[ip]
    while timestamps and timestamps[0] < window_start:
        timestamps.pop(0)
    if len(timestamps) >= CHAT_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"请求太频繁，每个IP每{CHAT_RATE_WINDOW_SECONDS}秒最多{CHAT_RATE_LIMIT}次提问，请稍后再试。",
        )
    timestamps.append(now)

BASE_DIR = os.path.dirname(__file__)
GEOJSON_PATH = os.path.join(BASE_DIR, "data", "grid_data.geojson")
WEIBO_JSON_PATH = os.path.join(BASE_DIR, "data", "weibo_posts.json")
WEIBO_EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "weibo_embeddings.npy")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

# 与 weibo_embed.py 用同一个模型名，保证查询向量和离线索引向量在同一空间里。
# 部署环境（Render免费档512MB内存）装不下本地PyTorch+transformers（光加载就要吃掉800MB+），
# 所以查询文本的embedding改成调用HuggingFace官方免费的Inference API远程算，
# 离线批量算索引向量（weibo_embed.py）仍然在本地机器上跑，两边模型一致，向量空间兼容。
WEIBO_EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{WEIBO_EMBED_MODEL_NAME}"
HF_TOKEN = os.environ.get("HF_TOKEN")
# 语义相关性阈值：低于此余弦相似度的帖子不算入候选池。
# 针对bge-small-zh-v1.5在这批微博短文本上的相似度分布实测校准得出，见任务10验证记录。
WEIBO_SIMILARITY_THRESHOLD = 0.5


def _embed_query_via_hf(text: str) -> np.ndarray:
    """调HuggingFace Inference API算查询文本的embedding，失败时抛异常由调用方处理"""
    if not HF_TOKEN:
        raise RuntimeError("未配置HF_TOKEN环境变量，无法调用HuggingFace Inference API")
    resp = requests.post(
        HF_INFERENCE_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=30,
    )
    resp.raise_for_status()
    vec = np.array(resp.json(), dtype=np.float32)
    # 有些模型返回[1, dim]或按token返回[seq_len, dim]，统一压成单个句向量
    while vec.ndim > 1:
        vec = vec.mean(axis=0)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

app = FastAPI(title="武汉城市活力地图")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

with open(GEOJSON_PATH, encoding="utf-8") as f:
    _GEOJSON = json.load(f)

_ROWS = []
for feat in _GEOJSON["features"]:
    row = dict(feat["properties"])
    row["lng"], row["lat"] = feat["geometry"]["coordinates"]
    _ROWS.append(row)
DF = pd.DataFrame(_ROWS)
KNOWN_DISTRICTS = sorted(DF["district"].dropna().unique().tolist())
GRID_ID_TO_DISTRICT = dict(zip(DF["grid_id"], DF["district"]))

PRED_COLS = [c for c in DF.columns if c.startswith("pred_")]

with open(WEIBO_JSON_PATH, encoding="utf-8") as f:
    WEIBO_DF = pd.DataFrame(json.load(f))
# place_type/post_time缺失时pandas读入会变成float NaN，FastAPI的JSONResponse默认
# allow_nan=False，序列化NaN会直接抛ValueError导致500。这里统一转回None(->JSON null)。
WEIBO_DF["place_type"] = WEIBO_DF["place_type"].astype(object).where(WEIBO_DF["place_type"].notna(), None)
WEIBO_DF["post_time"] = WEIBO_DF["post_time"].astype(object).where(WEIBO_DF["post_time"].notna(), None)

WEIBO_EMBEDDINGS = np.load(WEIBO_EMBEDDINGS_PATH)
assert len(WEIBO_EMBEDDINGS) == len(WEIBO_DF), "weibo_embeddings.npy行数与weibo_posts.json条数不一致，需要重新跑weibo_embed.py"


class ChatRequest(BaseModel):
    question: str


@app.get("/api/geojson")
def get_geojson():
    return JSONResponse(content=_GEOJSON)


@app.get("/api/districts")
def get_districts():
    return {"districts": KNOWN_DISTRICTS}


@app.get("/api/search")
def search(district: str | None = None, period: str | None = None,
           topn: int = 20, order: str = "desc"):
    """
    district: 行政区名（可选，子串匹配）
    period: label_col原始列名（如"工作日_夜间"），对应 pred_<period> 列；缺省用全部10列均值
    topn: 返回条数
    order: asc | desc
    """
    sub = DF
    if district:
        sub = sub[sub["district"].str.contains(district, na=False)]

    ascending = order == "asc"
    if ascending:
        # 最低活力排名不统计水域为主的格网（水域本身没有活力语义，排进"最低"没有意义）
        sub = sub[~sub["is_water"]]

    if period and f"pred_{period}" in sub.columns:
        value_col = f"pred_{period}"
        sub = sub.assign(_value=sub[value_col])
    else:
        sub = sub.assign(_value=sub[PRED_COLS].mean(axis=1))

    sub = sub.sort_values("_value", ascending=ascending).head(topn)

    cols = ["grid_id", "district", "lng", "lat", "_value", "missing_weibo", "missing_streetview"]
    result = sub[cols].rename(columns={"_value": "value"}).to_dict(orient="records")
    return {"count": len(result), "results": result}


def _resolve_periods_and_rows(intent: dict):
    periods = intent.get("periods") or []
    district = intent.get("district")
    direction = intent.get("direction")

    sub = DF
    if district:
        sub = sub[sub["district"].str.contains(district, na=False)]

    ascending = direction == "low"
    if ascending:
        sub = sub[~sub["is_water"]]

    if periods:
        value_col = f"pred_{periods[0]}"
    else:
        value_col = None
    sub = sub.assign(val=sub[value_col] if value_col else sub[PRED_COLS].mean(axis=1))

    sub = sub.sort_values("val", ascending=ascending).head(10)

    period_label = periods[0] if periods else "综合时段"
    rows = [
        {"grid_id": int(r.grid_id), "district": r.district, "period": period_label, "value": float(r.val)}
        for r in sub.itertuples()
    ]
    return rows


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    intent = retrieval.parse_intent(req.question, KNOWN_DISTRICTS)
    has_signal = bool(intent["periods"] or intent["district"] or intent["direction"])

    if not has_signal:
        answer = llm_client.answer_question(req.question, {"fallback": True})
        return {"answer": answer, "highlight_grid_ids": []}

    rows = _resolve_periods_and_rows(intent)
    context = {"intent": intent, "rows": rows}
    answer = llm_client.answer_question(req.question, context)
    return {"answer": answer, "highlight_grid_ids": [r["grid_id"] for r in rows]}


WEIBO_POST_COLS = ["text", "lng", "lat", "grid_id", "place_type", "post_time", "like_count"]


@app.get("/api/weibo/search")
def weibo_search(keyword: str, top_n: int = 150):
    """
    语义向量检索：查询文本经HuggingFace Inference API编码后与全部帖子embedding算
    余弦相似度，相似度超过WEIBO_SIMILARITY_THRESHOLD的算作候选池（不再靠字面关键词
    匹配，也不再受固定关键词数量限制）；候选池内按like_count（点赞数）降序取前top_n条。
    不调用DeepSeek，不受_check_rate_limit限流，但会受HuggingFace免费API自身的频率限制。
    """
    try:
        query_vec = _embed_query_via_hf(keyword)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"语义检索服务暂时不可用，请稍后再试。（{e}）")

    similarities = WEIBO_EMBEDDINGS.astype(np.float32) @ query_vec

    candidate_mask = similarities >= WEIBO_SIMILARITY_THRESHOLD
    candidates = WEIBO_DF[candidate_mask].copy()
    candidates["similarity"] = similarities[candidate_mask]

    total_relevant = len(candidates)
    top = candidates.sort_values("like_count", ascending=False).head(top_n)

    return {
        "total_relevant": total_relevant,
        "returned": len(top),
        "posts": top[WEIBO_POST_COLS + ["similarity"]].to_dict(orient="records"),
    }


@app.get("/api/weibo/grid/{grid_id}")
def weibo_grid_activity(grid_id: int, request: Request):
    """某格网内的脱敏微博样本 + 地点类型分布 + LLM活动解读"""
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    posts = WEIBO_DF[WEIBO_DF["grid_id"] == grid_id].sort_values("post_time", ascending=False)
    place_type_counts = (
        posts["place_type"].dropna().value_counts().head(10).items()
    )
    place_type_counts = [(name, int(count)) for name, count in place_type_counts]

    district = GRID_ID_TO_DISTRICT.get(grid_id, "未知区域")
    sample_texts = posts["text"].head(15).tolist()
    summary = llm_client.summarize_grid_activity(district, place_type_counts, sample_texts)

    return {
        "grid_id": grid_id,
        "count": len(posts),
        "place_type_stats": place_type_counts,
        "summary": summary,
        "posts": posts[["text", "place_type", "post_time"]].head(20).to_dict(orient="records"),
    }


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

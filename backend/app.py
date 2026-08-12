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
from fastapi.middleware.gzip import GZipMiddleware
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

# 与 weibo_embed.py 用同一个SiliconFlow embeddings模型，保证查询向量和离线索引
# 向量在同一空间里。部署环境（Render免费档512MB内存）装不下本地PyTorch+transformers
# （光加载就要吃掉800MB+），所以查询和离线索引都改成调用同一个云端embeddings API，
# 而不是本地/远程混用不同的服务——之前试过HuggingFace免费推理接口，它返回的是
# 没做池化的逐token向量，跟本地sentence-transformers算出来的向量对不上，检索
# 结果是垃圾数据；SiliconFlow的embeddings接口直接返回池化好的向量，两边统一
# 用它，从根上保证一致。
WEIBO_EMBED_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
SILICONFLOW_EMBEDDINGS_URL = "https://api.siliconflow.cn/v1/embeddings"
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
# 语义相关性阈值：低于此余弦相似度的帖子不算入候选池。
# 针对bge-large-zh-v1.5在这批微博短文本上的相似度分布实测校准得出。
WEIBO_SIMILARITY_THRESHOLD = 0.5

# 二阶段cross-encoder重排序：bi-encoder(embedding)召回的候选池里，query和document
# 是分别独立编码再比向量，两者在模型内部从未"见过面"，排序精度有限（高赞但字面
# 沾边的帖子容易靠点赞数挤到前面）。reranker模型把query和每条候选文本拼在一起
# 一次性输入，做联合attention后打一个直接的相关性分数，精度明显更高，是2025-2026
# 生产级RAG系统的标配二阶段。SiliconFlow提供Cohere兼容的/v1/rerank接口，同一账号
# 直接可用，不需要额外引入服务商或本地模型。
WEIBO_RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
SILICONFLOW_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
# reranker是逐对query-document做联合推理，比embedding的向量点积慢得多，不能对
# 整个候选池（可能上万条）都跑一遍。精排范围要覆盖实际展示的top_n条，否则
# 展示列表里排在精排范围之外的部分会退回到"相似度×点赞数"初筛排序，让高赞但
# 弱相关的帖子重新露出来（之前设成固定50条时，top_n=150的默认展示第51~150名
# 全部漏了精排，就是这么踩的坑）。RERANK_MAX_CANDIDATES是防止极端大top_n请求
# 单次调用文档数过多的安全上限，超过这个上限的尾部才退回初筛排序。
RERANK_MAX_CANDIDATES = 200


def _embed_query(text: str) -> np.ndarray:
    """调SiliconFlow embeddings API算查询文本的embedding，失败时抛异常由调用方处理"""
    if not SILICONFLOW_API_KEY:
        raise RuntimeError("未配置SILICONFLOW_API_KEY环境变量，无法调用SiliconFlow embeddings API")
    resp = requests.post(
        SILICONFLOW_EMBEDDINGS_URL,
        headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
        json={"model": WEIBO_EMBED_MODEL_NAME, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    vec = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _rerank_scores(query: str, documents: list[str]) -> list[float]:
    """调SiliconFlow rerank API给query-document对打联合相关性分数，按documents原始顺序返回分数列表"""
    resp = requests.post(
        SILICONFLOW_RERANK_URL,
        headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
        json={"model": WEIBO_RERANK_MODEL_NAME, "query": query, "documents": documents},
        timeout=30,
    )
    resp.raise_for_status()
    scores = [0.0] * len(documents)
    for item in resp.json()["results"]:
        scores[item["index"]] = item["relevance_score"]
    return scores

app = FastAPI(title="武汉城市活力地图")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
# 全量POI接口一次性传约9MB JSON，开GZip压缩能大幅减小实际传输体积，减轻Render免费档带宽压力。
app.add_middleware(GZipMiddleware, minimum_size=1000)

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

# 全量POI地图浏览用的轻量字段列表，启动时算一次存着，避免每次请求都重新from_dict转换
# 9.7万行。不带原文（隐私+体积考虑，9.7万条全文会让单次响应膨胀到近30MB），点位详情
# 复用已有的/api/weibo/grid/{grid_id}按需拉取。
ALL_POI_LIGHT_COLS = ["lng", "lat", "grid_id", "place_type", "like_count"]
ALL_POI_LIGHT = WEIBO_DF[ALL_POI_LIGHT_COLS].to_dict(orient="records")


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
    语义向量检索，两阶段：
    1. 召回：查询文本经SiliconFlow embeddings API编码后与全部帖子embedding算余弦相似度，
       相似度超过WEIBO_SIMILARITY_THRESHOLD的算作候选池（不靠字面关键词匹配）。候选池
       内先按初筛分数（相似度×log(1+点赞数)）排序，避免"很火但不太相关"的帖子靠人气
       霸榜，取前min(top_n, RERANK_MAX_CANDIDATES)名进入精排（覆盖实际展示条数）。
    2. 精排：query和候选文本一起送SiliconFlow rerank API（cross-encoder联合attention），
       按真实相关性分数重新排序；候选池里没进精排的其余帖子仍按初筛分数接在后面。
    精排调用失败（如未触发限流之外的异常）会静默降级为只用初筛分数排序，不影响整体检索可用性。
    不调用DeepSeek，不受_check_rate_limit限流，但会受SiliconFlow API自身的频率限制。
    """
    try:
        query_vec = _embed_query(keyword)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"语义检索服务暂时不可用，请稍后再试。（{e}）")

    # 注意：不要把WEIBO_EMBEDDINGS(float16,9.7万行)整体转float32——那会在每次请求时
    # 临时多分配约190MB内存，在Render免费档512MB内存上很容易把进程冲爆(502)。
    # 反过来把很小的query_vec转成float16去匹配，只需要几KB。
    similarities = WEIBO_EMBEDDINGS @ query_vec.astype(np.float16)
    similarities = similarities.astype(np.float32)

    candidate_mask = similarities >= WEIBO_SIMILARITY_THRESHOLD
    candidates = WEIBO_DF[candidate_mask].copy()
    candidates["similarity"] = similarities[candidate_mask]

    total_relevant = len(candidates)

    candidates["score"] = candidates["similarity"] * np.log1p(candidates["like_count"])
    candidates = candidates.sort_values("score", ascending=False)

    shortlist_n = min(top_n, len(candidates), RERANK_MAX_CANDIDATES)
    shortlist = candidates.head(shortlist_n).copy()
    rest = candidates.iloc[shortlist_n:]

    try:
        rerank_scores = _rerank_scores(keyword, shortlist["text"].tolist())
        shortlist["score"] = rerank_scores
        shortlist = shortlist.sort_values("score", ascending=False)
    except Exception as e:
        print(f"重排序失败，降级为仅用初筛分数排序：{e}", flush=True)

    top = pd.concat([shortlist, rest]).head(top_n)

    return {
        "total_relevant": total_relevant,
        "returned": len(top),
        "posts": top[WEIBO_POST_COLS + ["similarity"]].to_dict(orient="records"),
    }


@app.get("/api/weibo/all_pois")
def get_all_pois():
    """
    全量微博POI点位（轻量字段：经纬度/格网/类别/点赞数，不含原文），供地图"全部POI"
    浏览模式一次性拉取后前端聚合渲染+按类别/点赞数纯前端筛选，不受_check_rate_limit限流。
    """
    return {"count": len(ALL_POI_LIGHT), "posts": ALL_POI_LIGHT}


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

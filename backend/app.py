# ==============================================================
#  FastAPI 主服务：地图数据 / 查询检索 / LLM问答(mock) / 前端静态文件
# ==============================================================

import itertools
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agent_client
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
STUDY_AREA_BOUNDARY_PATH = os.path.join(BASE_DIR, "data", "study_area_boundary.geojson")
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

# 和风天气：新版平台按项目分配专属API Host（不是共享的devapi.qweather.com了，
# 裸用共享域名会被拒403 Invalid Host），host要去控制台"设置"页面查自己项目的。
# 免费未认证账号只能拿/v7/weather/3d（未来3天，含今天）；认证开发者账号才能用
# /v7/weather/7d拿7天，到时候只要改QWEATHER_FORECAST_DAYS就行，不用改调用逻辑。
# 武汉的LocationID是固定值，用和风天气GeoAPI查过。
QWEATHER_API_KEY = os.environ.get("QWEATHER_API_KEY")
QWEATHER_HOST = "https://p26vhhq5qq.re.qweatherapi.com"
QWEATHER_FORECAST_DAYS = "3d"
WUHAN_LOCATION_ID = "101200101"

# Tavily网页搜索：给agent当"兜底"工具用，覆盖微博数据集/城市活力模型都查不到的
# 开放性信息。认证方式是Authorization:Bearer头（实测确认过，官方文档给的Python
# SDK示例容易让人以为要走别的认证方式）。
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
TAVILY_API_URL = "https://api.tavily.com/search"

# 高德地图Web服务API：地理编码（地名->经纬度）+ 路径规划（驾车/步行/公交含地铁）。
# 注意申请key时要选"Web服务"平台，不是"Web端(JS API)"——后者是给网页里嵌交互式
# 地图组件用的，我们这里是后端直接发HTTP请求调REST接口，跟嵌入式地图组件无关。
AMAP_API_KEY = os.environ.get("AMAP_API_KEY")
AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_DIRECTION_URLS = {
    "driving": "https://restapi.amap.com/v3/direction/driving",
    "walking": "https://restapi.amap.com/v3/direction/walking",
    "transit": "https://restapi.amap.com/v3/direction/transit/integrated",
}

# 中国法定节假日+调休数据：用NateScarlet/holiday-cn这个社区维护的开源数据集（每日
# 自动抓取国务院公告更新），比自己按"周六周日=休息"简单判断准确——调休补班日
# （比如国庆调休的周末上班）在这份数据里会被显式标成isOffDay=false。
# 按年缓存在内存里，一年最多请求一次GitHub，避免每次查日期都重新拉取。
HOLIDAY_CN_URL_TEMPLATE = "https://raw.githubusercontent.com/NateScarlet/holiday-cn/master/{year}.json"
_holiday_cache: dict[int, dict] = {}  # year -> {date_str: {"name":..., "is_off_day": bool}}


def _get_holiday_map(year: int) -> dict:
    if year in _holiday_cache:
        return _holiday_cache[year]
    try:
        resp = requests.get(HOLIDAY_CN_URL_TEMPLATE.format(year=year), timeout=10)
        resp.raise_for_status()
        days = resp.json()["days"]
        holiday_map = {d["date"]: {"name": d["name"], "is_off_day": d["isOffDay"]} for d in days}
    except Exception as e:
        # 拉取失败不缓存空结果——只是这次查询降级为仅按周末判断，下次请求还会重试，
        # 避免一次网络抖动就让这一整年永久锁死在"没有节假日数据"的错误状态。
        print(f"节假日数据拉取失败（{year}年），本次查询降级为仅按周末判断：{e}", flush=True)
        return {}
    _holiday_cache[year] = holiday_map
    return holiday_map


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

with open(STUDY_AREA_BOUNDARY_PATH, encoding="utf-8") as f:
    _STUDY_AREA_BOUNDARY = json.load(f)

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
    # question+history：全新一轮（或者上一轮已正常结束、紧接着问的新问题）。
    # history是前端本地压缩存好的最近几轮{question,answer}，不传等价于没有记忆。
    # pending_turn+reply：上一轮被ask_user打断、这次是恢复——pending_turn是上次
    # "ask_user"事件里原样发给前端、又原样传回来的快照，不做任何加工，直接透传给
    # agent_client.run_agent_stream。两种方式二选一，都不传时question按空字符串处理。
    question: str | None = None
    history: list[dict] | None = None
    pending_turn: dict | None = None
    reply: str | None = None


@app.get("/api/geojson")
def get_geojson():
    return JSONResponse(content=_GEOJSON)


@app.get("/api/study_area_boundary")
def get_study_area_boundary():
    """三环内研究区域的整体边界线（单个多边形轮廓，不是逐格网边框），
    数据来自原始数据/shp文件/武汉三环面.shp转出的GeoJSON。"""
    return JSONResponse(content=_STUDY_AREA_BOUNDARY)


@app.get("/api/districts")
def get_districts():
    return {"districts": KNOWN_DISTRICTS}


@app.get("/api/calendar/day_type")
def get_day_type(date_str: str):
    """
    判断某天是"工作日"还是"休息日"，含法定节假日/调休（数据源见_get_holiday_map）。
    date_str格式YYYY-MM-DD。节假日数据里没有的日期，退回"周一到周五=工作日，
    周六周日=休息日"的默认规则。
    """
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式应为YYYY-MM-DD")

    holiday_map = _get_holiday_map(d.year)
    entry = holiday_map.get(date_str)
    if entry is not None:
        is_workday = not entry["is_off_day"]
        note = entry["name"] + ("调休上班" if is_workday else "")
    else:
        is_workday = d.weekday() < 5  # 0=周一…4=周五
        note = None

    return {
        "date": date_str,
        "is_workday": is_workday,
        "label": "工作日" if is_workday else "休息日",
        "note": note,
    }


@app.get("/api/weather")
def get_weather(date_str: str):
    """
    查武汉某天的天气预报。免费未认证账号只能查未来3天（含今天），超出范围会
    返回404并提示原因，而不是笼统的"查询失败"。
    """
    if not QWEATHER_API_KEY:
        raise HTTPException(status_code=503, detail="未配置QWEATHER_API_KEY环境变量，无法查询天气")
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式应为YYYY-MM-DD")

    try:
        resp = requests.get(
            f"{QWEATHER_HOST}/v7/weather/{QWEATHER_FORECAST_DAYS}",
            params={"location": WUHAN_LOCATION_ID},
            headers={"X-QW-Api-Key": QWEATHER_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"天气服务暂时不可用，请稍后再试。（{e}）")

    if payload.get("code") != "200":
        raise HTTPException(status_code=503, detail=f"和风天气接口返回异常（code={payload.get('code')}）")

    for day in payload.get("daily", []):
        if day["fxDate"] == date_str:
            return {
                "date": date_str,
                "text_day": day["textDay"],
                "text_night": day["textNight"],
                "temp_min": day["tempMin"],
                "temp_max": day["tempMax"],
            }

    raise HTTPException(
        status_code=404,
        detail=f"仅支持查询未来{QWEATHER_FORECAST_DAYS[:-1]}天的天气（免费未认证账号限制），{date_str}超出范围",
    )


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


# "问答"和"智能推荐"原来是两个独立入口，普通用户分不清该用哪个。合并成一个
# /api/chat：parse_intent能命中时（时段/行政区/高低方向都是关键词能可靠解析的
# 封闭式问题）走快速路径（单次DeepSeek调用），命不中、或问题带这些开放性信号词
# （需要多步骤查天气/路线/热点才能回答）时才走agent循环（慢/贵，但SSE流式返回
# 让用户能看到中间步骤，不会像纯等待一样没有反馈）。
AGENT_TRIGGER_KEYWORDS = ["推荐", "路线", "规划", "去哪", "怎么玩", "一日游", "行程", "打卡", "周末"]

# 快速路径是纯关键词解析+单次数据库查询，完全不知道"历史对话"这回事——如果问题里
# 恰好也命中了period/district/direction关键词（哪怕本意是追问上一轮，比如"你刚才说的
# 活力最高那个格网"里的"活力""高"会命中direction关键词），会被判定has_signal=True走
# 快速路径，安安静静地重新查一次全新数据，跟上一轮答案对不上却不会报错——这是真实
# 复现过的bug，不是假设：只要问题里带这类"指代之前对话"的词、且这次请求确实带了历史，
# 就强制走agent路径（有历史记忆能力），不看关键词解析结果如何。
MEMORY_REFERENCE_KEYWORDS = ["刚才", "刚刚", "刚说", "上面", "上一条", "上次", "之前", "你说", "提到的"]


def _should_use_agent(question: str, intent: dict, has_history: bool = False) -> bool:
    has_signal = bool(intent["periods"] or intent["district"] or intent["direction"])
    has_open_ended_kw = any(kw in question for kw in AGENT_TRIGGER_KEYWORDS)
    references_memory = has_history and any(kw in question for kw in MEMORY_REFERENCE_KEYWORDS)
    return has_open_ended_kw or not has_signal or references_memory


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    """
    统一问答入口，SSE流式返回。快速路径只发一个"final"事件（跟agent路径的最终
    事件同格式），前端不用区分走的是哪条路径，都当成同一套事件流处理；agent路径
    会先发若干"tool_call"/"tool_result"事件展示中间步骤，中途可能发一个"ask_user"
    事件暂停等用户回答，最终发"final"事件。

    pending_turn有值时，说明这是在恢复一次被ask_user打断的对话——直接进agent循环
    续接，不再走parse_intent那套快速路径判断（快速路径设计上就是处理独立的、封闭式
    的单轮问题，跟"接着上次没问完的话题继续"这件事没关系）。
    """
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    def event_stream():
        if req.pending_turn:
            for event in agent_client.run_agent_stream(
                AGENT_TOOL_IMPLS, _extract_map_features,
                pending_turn=req.pending_turn, reply=req.reply,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            return

        question = req.question or ""
        intent = retrieval.parse_intent(question, KNOWN_DISTRICTS)
        use_agent = _should_use_agent(question, intent, bool(req.history))

        if use_agent:
            for event in agent_client.run_agent_stream(
                AGENT_TOOL_IMPLS, _extract_map_features,
                question=question, history=req.history,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            return

        has_signal = bool(intent["periods"] or intent["district"] or intent["direction"])
        if has_signal:
            rows = _resolve_periods_and_rows(intent)
            context = {"intent": intent, "rows": rows}
            answer = llm_client.answer_question(question, context)
            highlight_grid_ids = [r["grid_id"] for r in rows]
        else:
            answer = llm_client.answer_question(question, {"fallback": True})
            highlight_grid_ids = []
        event = {"type": "final", "answer": answer, "highlight_grid_ids": highlight_grid_ids}
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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


# ==============================================================
#  智能推荐agent（案例一/二）：反复调用DeepSeek + 下面这几个工具，直到给出
#  最终答案。循环编排逻辑在agent_client.py，这里只负责"工具具体怎么执行"——
#  全部直接复用上面已经写好的路由函数，不重新实现一遍逻辑。返回给模型的数据
#  都做了精简（截断文本、限制条数、去掉经纬度这类模型用不上的字段），控制
#  token成本。
# ==============================================================

def _tool_is_workday(date: str) -> dict:
    try:
        return get_day_type(date_str=date)
    except HTTPException as e:
        return {"error": e.detail}


def _tool_get_weather(date: str) -> dict:
    try:
        return get_weather(date_str=date)
    except HTTPException as e:
        return {"error": e.detail}


def _tool_get_vitality(period: str | None = None, district: str | None = None,
                        order: str = "desc", topn: int = 10) -> dict:
    result = search(district=district, period=period, topn=topn, order=order)
    return {
        "count": result["count"],
        "results": [
            {"grid_id": r["grid_id"], "district": r["district"], "value": round(r["value"], 2)}
            for r in result["results"]
        ],
    }


def _tool_search_weibo_hotspots(keyword: str, top_n: int = 20) -> dict:
    result = weibo_search(keyword=keyword, top_n=top_n)
    posts = result["posts"][:10]  # 只给模型看前10条，控制token成本
    return {
        "total_relevant": result["total_relevant"],
        "posts": [
            {
                "text": (p["text"] or "")[:60],
                "grid_id": p["grid_id"],
                "district": GRID_ID_TO_DISTRICT.get(p["grid_id"], "未知"),
                "place_type": p["place_type"],
                "like_count": p["like_count"],
            }
            for p in posts
        ],
    }


def _tool_web_search(query: str) -> dict:
    if not TAVILY_API_KEY:
        return {"error": "未配置TAVILY_API_KEY环境变量，网页搜索不可用"}
    try:
        resp = requests.post(
            TAVILY_API_URL,
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
            json={"query": query, "search_depth": "basic", "max_results": 5},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": f"网页搜索失败：{e}"}
    # 实测Tavily基础检索(search_depth=basic)每条结果本身就有1200~1500字左右，
    # 之前截到200字太狠——像天气40天预报这种表格型页面，前200字大概率还停在
    # 导航菜单，真正的逐日数据在更靠后的位置，截断反而把有用内容漏掉了，导致
    # 模型明明搜到了理论上覆盖到的页面，却因为看到的是被砍掉的片段而判断"查不到"。
    # 1200基本能覆盖Tavily basic模式的完整内容，不会显著增加token成本。
    return {
        "results": [
            {"title": r["title"], "content": (r["content"] or "")[:1200], "url": r["url"]}
            for r in data.get("results", [])[:5]
        ],
    }


def _tool_geocode(address: str) -> dict:
    """地名->精确经纬度，用高德地理编码API（city限定武汉，避免同名地点歧义）"""
    if not AMAP_API_KEY:
        return {"error": "未配置AMAP_API_KEY环境变量，无法查询地点坐标"}
    try:
        resp = requests.get(
            AMAP_GEOCODE_URL,
            params={"address": address, "city": "武汉", "key": AMAP_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": f"地理编码查询失败：{e}"}
    if data.get("status") != "1" or not data.get("geocodes"):
        return {"error": f"没查到「{address}」的坐标，换个更精确/更常见的地名试试"}
    geo = data["geocodes"][0]
    lng, lat = geo["location"].split(",")
    return {
        "name": address,
        "formatted_address": geo.get("formatted_address"),
        "lng": float(lng),
        "lat": float(lat),
    }


def _parse_amap_polyline(polyline_str: str) -> list[list[float]]:
    """高德polyline字符串"lng,lat;lng,lat;..."解析成[[lng,lat],...]坐标数组"""
    points = []
    for pair in polyline_str.split(";"):
        if not pair:
            continue
        lng_str, lat_str = pair.split(",")
        points.append([float(lng_str), float(lat_str)])
    return points


def _tool_route_between(origin_lng: float, origin_lat: float, dest_lng: float, dest_lat: float,
                         mode: str = "driving") -> dict:
    """
    两点间真实路线，接高德Direction API。mode: driving(驾车)/walking(步行)/
    transit(公交+地铁，含换乘站和地铁出入口信息)。返回给模型的字段只有精简摘要
    （距离/耗时/公交地铁线路名+上下车站+出入口）；真实道路坐标串存在"_polyline"
    这个下划线开头的字段里——agent_client.py喂给模型前会把下划线字段过滤掉
    （模型不需要几百个坐标点，那样只会浪费token），但会保留给前端画在地图上。
    """
    if not AMAP_API_KEY:
        return {"error": "未配置AMAP_API_KEY环境变量，无法查询路线"}
    url = AMAP_DIRECTION_URLS.get(mode)
    if not url:
        return {"error": f"不支持的出行方式：{mode}，可选driving/walking/transit"}

    params = {
        "origin": f"{origin_lng},{origin_lat}",
        "destination": f"{dest_lng},{dest_lat}",
        "key": AMAP_API_KEY,
    }
    if mode == "transit":
        params["city"] = "武汉"

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": f"路线查询失败：{e}"}

    if data.get("status") != "1":
        return {"error": f"高德接口返回异常：{data.get('info')}"}

    if mode in ("driving", "walking"):
        paths = data.get("route", {}).get("paths", [])
        if not paths:
            return {"error": "查不到这两点间的路线"}
        path = paths[0]
        polyline = []
        for step in path.get("steps", []):
            if step.get("polyline"):
                polyline.extend(_parse_amap_polyline(step["polyline"]))
        return {
            "mode": mode,
            "distance_m": int(path["distance"]),
            "duration_min": round(int(path["duration"]) / 60, 1),
            "_polyline": polyline,
        }

    # transit（公交/地铁组合）：取推荐的第一个换乘方案，逐段摘出关键信息
    transits = data.get("route", {}).get("transits", [])
    if not transits:
        return {"error": "查不到这两点间的公交/地铁路线"}
    t = transits[0]
    segments = []
    polyline = []
    for seg in t.get("segments", []):
        walking = seg.get("walking")
        if walking and walking.get("steps"):
            for step in walking["steps"]:
                if step.get("polyline"):
                    polyline.extend(_parse_amap_polyline(step["polyline"]))

        buslines = seg.get("bus", {}).get("buslines")
        if buslines:
            line = buslines[0]
            entry = {
                "type": "地铁" if "地铁" in (line.get("type") or "") else "公交",
                "line_name": line.get("name"),
                "from_stop": line.get("departure_stop", {}).get("name"),
                "to_stop": line.get("arrival_stop", {}).get("name"),
            }
            if seg.get("entrance", {}).get("name"):
                entry["entrance"] = seg["entrance"]["name"]
            if seg.get("exit", {}).get("name"):
                entry["exit"] = seg["exit"]["name"]
            segments.append(entry)
            if line.get("polyline"):
                polyline.extend(_parse_amap_polyline(line["polyline"]))
        elif walking and walking.get("distance"):
            segments.append({"type": "步行", "distance_m": int(walking["distance"])})

    return {
        "mode": "transit",
        "total_duration_min": round(int(t["duration"]) / 60, 1),
        "walking_distance_m": int(t.get("walking_distance", 0)),
        "segments": segments,
        "_polyline": polyline,
    }


def _haversine_m(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _tool_plan_route_order(points: list[dict]) -> dict:
    """
    points: [{"name":..., "lng":..., "lat":...}, ...]
    给多个候选点排一个访问顺序，纯本地计算、不调外部API：用直线距离暴力枚举
    全排列，找总距离最短的顺序（不是精确路网距离，只用来决定"先去哪后去哪"这个
    大致顺序，真实路线交给route_between逐段查）。8个点以内全排列（8!=4万级，
    毫秒级跑完）；超过8个退化成贪心最近邻，避免排列组合数爆炸。
    """
    n = len(points)
    if n < 2:
        return {"error": "至少需要2个点才能规划访问顺序"}

    def total_distance(order):
        return sum(
            _haversine_m(order[i]["lng"], order[i]["lat"], order[i + 1]["lng"], order[i + 1]["lat"])
            for i in range(len(order) - 1)
        )

    if n <= 8:
        best_order = list(min(itertools.permutations(points), key=total_distance))
    else:
        remaining = points[1:]
        best_order = [points[0]]
        while remaining:
            last = best_order[-1]
            nxt = min(remaining, key=lambda p: _haversine_m(last["lng"], last["lat"], p["lng"], p["lat"]))
            best_order.append(nxt)
            remaining.remove(nxt)

    return {
        "order": [p["name"] for p in best_order],
        "points": best_order,
        "total_straight_line_distance_m": round(total_distance(best_order)),
    }


AGENT_TOOL_IMPLS = {
    "is_workday": _tool_is_workday,
    "get_weather": _tool_get_weather,
    "get_vitality": _tool_get_vitality,
    "search_weibo_hotspots": _tool_search_weibo_hotspots,
    "web_search": _tool_web_search,
    "geocode": _tool_geocode,
    "route_between": _tool_route_between,
    "plan_route_order": _tool_plan_route_order,
}


def _extract_map_features(tool_name: str, result: dict) -> dict:
    """
    从工具结果里挑出能在地图上画出来的东西：
    - markers：geocode查到的地点，前端画点用
    - polylines：route_between查到的真实道路坐标串（存在"_polyline"字段里，
      不会被喂给模型，只在这里、只给地图用）
    - seen_grid_ids：get_vitality/search_weibo_hotspots这次结果里出现过的格网编号。
      **不会自动展示在地图上**——格网高亮只来自模型显式调用finish(answer,
      highlight_grid_ids)时主动挑选的，见agent_client.run_agent_stream里的说明。
      seen_grid_ids只是agent_client用来做白名单校验的事实依据：模型在finish里选的
      highlight_grid_ids必须是这一路真实查询到过的格网子集，防止编造/记错一个从没
      查到过的编号却被原样显示到地图上。这是确定性的事实核查（"是不是真查到过"），
      不影响"该不该展示"这个模型自己的判断。
    """
    if tool_name == "geocode" and "lng" in result and "lat" in result:
        return {"markers": [{"name": result.get("name"), "lng": result["lng"], "lat": result["lat"]}]}
    if tool_name == "route_between" and result.get("_polyline"):
        return {"polylines": [result["_polyline"]]}
    if tool_name == "get_vitality":
        return {"seen_grid_ids": [r["grid_id"] for r in result.get("results", [])]}
    if tool_name == "search_weibo_hotspots":
        return {"seen_grid_ids": [p["grid_id"] for p in result.get("posts", [])]}
    return {}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

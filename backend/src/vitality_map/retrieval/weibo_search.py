# ==============================================================
#  微博语义检索：embedding召回 + cross-encoder重排序两阶段RAG流程。
#  原来是app.py里的模块级函数+路由，现在拆成"检索逻辑"这一层，供
#  api/weibo.py（路由）和tools/weibo.py（agent工具）两边共用同一份实现。
# ==============================================================

import numpy as np
import pandas as pd
import requests
from fastapi import HTTPException

from vitality_map.core.config import settings
from vitality_map.core.data import WEIBO_DF, WEIBO_EMBEDDINGS

WEIBO_POST_COLS = ["text", "lng", "lat", "grid_id", "place_type", "post_time", "like_count"]


def _embed_query(text: str) -> np.ndarray:
    """
    调SiliconFlow embeddings API算查询文本的embedding，失败时抛异常由调用方处理。

    与 scripts/weibo_embed.py 用同一个SiliconFlow embeddings模型，保证查询向量和
    离线索引向量在同一空间里——之前试过HuggingFace免费推理接口，它返回的是没做
    池化的逐token向量，跟本地sentence-transformers算出来的向量对不上，检索结果是
    垃圾数据；SiliconFlow的embeddings接口直接返回池化好的向量，两边统一用它，
    从根上保证一致。
    """
    if not settings.siliconflow_api_key:
        raise RuntimeError("未配置SILICONFLOW_API_KEY环境变量，无法调用SiliconFlow embeddings API")
    resp = requests.post(
        settings.siliconflow_embeddings_url,
        headers={"Authorization": f"Bearer {settings.siliconflow_api_key}"},
        json={"model": settings.weibo_embed_model_name, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    vec = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _rerank_scores(query: str, documents: list[str]) -> list[float]:
    """
    调SiliconFlow rerank API给query-document对打联合相关性分数，按documents原始
    顺序返回分数列表。二阶段cross-encoder重排序：bi-encoder(embedding)召回的候选池
    里，query和document是分别独立编码再比向量，两者在模型内部从未"见过面"，排序
    精度有限（高赞但字面沾边的帖子容易靠点赞数挤到前面）。reranker模型把query和
    每条候选文本拼在一起一次性输入，做联合attention后打一个直接的相关性分数，
    精度明显更高，是2025-2026生产级RAG系统的标配二阶段。
    """
    resp = requests.post(
        settings.siliconflow_rerank_url,
        headers={"Authorization": f"Bearer {settings.siliconflow_api_key}"},
        json={"model": settings.weibo_rerank_model_name, "query": query, "documents": documents},
        timeout=30,
    )
    resp.raise_for_status()
    scores = [0.0] * len(documents)
    for item in resp.json()["results"]:
        scores[item["index"]] = item["relevance_score"]
    return scores


def weibo_search(keyword: str, top_n: int = 150) -> dict:
    """
    语义向量检索，两阶段：
    1. 召回：查询文本经SiliconFlow embeddings API编码后与全部帖子embedding算余弦
       相似度，相似度超过weibo_similarity_threshold的算作候选池（不靠字面关键词
       匹配）。候选池内先按初筛分数（相似度×log(1+点赞数)）排序，避免"很火但不
       太相关"的帖子靠人气霸榜，取前min(top_n, rerank_max_candidates)名进入精排
       （覆盖实际展示条数）。
    2. 精排：query和候选文本一起送SiliconFlow rerank API（cross-encoder联合
       attention），按真实相关性分数重新排序；候选池里没进精排的其余帖子仍按
       初筛分数接在后面。
    精排调用失败（如未触发限流之外的异常）会静默降级为只用初筛分数排序，不影响
    整体检索可用性。
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

    candidate_mask = similarities >= settings.weibo_similarity_threshold
    candidates = WEIBO_DF[candidate_mask].copy()
    candidates["similarity"] = similarities[candidate_mask]

    total_relevant = len(candidates)

    candidates["score"] = candidates["similarity"] * np.log1p(candidates["like_count"])
    candidates = candidates.sort_values("score", ascending=False)

    shortlist_n = min(top_n, len(candidates), settings.rerank_max_candidates)
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

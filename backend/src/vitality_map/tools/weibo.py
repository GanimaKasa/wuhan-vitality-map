# ==============================================================
#  微博相关业务逻辑：热点检索（agent工具包装）+ 单格网活动解读
# ==============================================================

from vitality_map.core.data import GRID_ID_TO_DISTRICT, WEIBO_DF
from vitality_map.retrieval.weibo_search import weibo_search
from vitality_map.services import llm_client


def tool_search_weibo_hotspots(keyword: str, top_n: int = 20) -> dict:
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


def get_grid_activity(grid_id: int) -> dict:
    """某格网内的脱敏微博样本 + 地点类型分布 + LLM活动解读"""
    posts = WEIBO_DF[WEIBO_DF["grid_id"] == grid_id].sort_values("post_time", ascending=False)
    place_type_counts = posts["place_type"].dropna().value_counts().head(10).items()
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

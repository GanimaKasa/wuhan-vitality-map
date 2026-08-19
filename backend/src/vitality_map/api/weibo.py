from fastapi import APIRouter, Request

from vitality_map.core.data import ALL_POI_LIGHT
from vitality_map.core.rate_limit import check_rate_limit
from vitality_map.retrieval.weibo_search import weibo_search
from vitality_map.tools.weibo import get_grid_activity

router = APIRouter(tags=["weibo"])


@router.get("/api/weibo/search")
def weibo_search_route(keyword: str, top_n: int = 150):
    """语义向量检索，不调用DeepSeek，不受check_rate_limit限流，但会受SiliconFlow
    API自身的频率限制。具体两阶段召回+精排逻辑见retrieval/weibo_search.py。"""
    return weibo_search(keyword=keyword, top_n=top_n)


@router.get("/api/weibo/all_pois")
def get_all_pois():
    """全量微博POI点位（轻量字段：经纬度/格网/类别/点赞数，不含原文），供地图
    "全部POI"浏览模式一次性拉取后前端聚合渲染+按类别/点赞数纯前端筛选，不受
    限流。"""
    return {"count": len(ALL_POI_LIGHT), "posts": ALL_POI_LIGHT}


@router.get("/api/weibo/grid/{grid_id}")
def weibo_grid_activity(grid_id: int, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)
    return get_grid_activity(grid_id)

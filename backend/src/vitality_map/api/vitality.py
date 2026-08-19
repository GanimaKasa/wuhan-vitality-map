from fastapi import APIRouter

from vitality_map.tools.vitality import search

router = APIRouter(tags=["vitality"])


@router.get("/api/search")
def search_route(district: str | None = None, period: str | None = None,
                  topn: int = 20, order: str = "desc"):
    return search(district=district, period=period, topn=topn, order=order)

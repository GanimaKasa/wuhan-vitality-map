from fastapi import APIRouter

from vitality_map.tools.calendar import get_day_type

router = APIRouter(tags=["calendar"])


@router.get("/api/calendar/day_type")
def day_type_route(date_str: str):
    return get_day_type(date_str)

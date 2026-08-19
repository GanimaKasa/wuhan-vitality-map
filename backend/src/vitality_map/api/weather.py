from fastapi import APIRouter

from vitality_map.tools.weather import get_weather

router = APIRouter(tags=["weather"])


@router.get("/api/weather")
def weather_route(date_str: str):
    return get_weather(date_str)

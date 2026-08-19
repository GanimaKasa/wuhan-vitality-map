# ==============================================================
#  天气查询（和风天气API）
# ==============================================================

from datetime import datetime

import requests
from fastapi import HTTPException

from vitality_map.core.config import settings


def get_weather(date_str: str) -> dict:
    """
    查武汉某天的天气预报。免费未认证账号只能查未来3天（含今天），超出范围会
    返回404并提示原因，而不是笼统的"查询失败"。
    """
    if not settings.qweather_api_key:
        raise HTTPException(status_code=503, detail="未配置QWEATHER_API_KEY环境变量，无法查询天气")
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式应为YYYY-MM-DD")

    try:
        resp = requests.get(
            f"{settings.qweather_host}/v7/weather/{settings.qweather_forecast_days}",
            params={"location": settings.wuhan_location_id},
            headers={"X-QW-Api-Key": settings.qweather_api_key},
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
        detail=f"仅支持查询未来{settings.qweather_forecast_days[:-1]}天的天气（免费未认证账号限制），{date_str}超出范围",
    )


def tool_get_weather(date: str) -> dict:
    """agent工具包装：把HTTPException转成模型能看懂的错误dict"""
    try:
        return get_weather(date_str=date)
    except HTTPException as e:
        return {"error": e.detail}

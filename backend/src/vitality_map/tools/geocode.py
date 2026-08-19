# ==============================================================
#  地理编码（地名 -> 经纬度），高德地图Web服务API
# ==============================================================

import requests

from vitality_map.core.config import settings


def tool_geocode(address: str) -> dict:
    """地名->精确经纬度，用高德地理编码API（city限定武汉，避免同名地点歧义）"""
    if not settings.amap_api_key:
        return {"error": "未配置AMAP_API_KEY环境变量，无法查询地点坐标"}
    try:
        resp = requests.get(
            settings.amap_geocode_url,
            params={"address": address, "city": "武汉", "key": settings.amap_api_key},
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

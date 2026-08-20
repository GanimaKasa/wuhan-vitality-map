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


def _parse_poi(poi: dict) -> dict:
    lng, lat = poi.get("location", ",").split(",") if poi.get("location") else (None, None)
    return {
        "name": poi.get("name"),
        "address": poi.get("address"),
        "type": poi.get("type"),
        "lng": float(lng) if lng else None,
        "lat": float(lat) if lat else None,
    }


def tool_search_poi(keyword: str, center_lng: float | None = None, center_lat: float | None = None,
                     radius: int = 3000, top_n: int = 10) -> dict:
    """
    按关键字搜索地点(POI)，用高德地图Web服务API：
    - 传了center_lng/center_lat：用"周边搜索"，在center为圆心、radius米范围内找（比如
      "黄鹤楼附近500米的餐厅"）
    - 没传中心点：用"关键字搜索"，city限定武汉全城范围（比如"武汉网红打卡地"）
    返回结构化的候选点列表(name/address/type/lng/lat)，比web_search+人工摘取地名更精确、
    更适合直接喂给geocode下游的plan_route_order/route_between。
    """
    if not settings.amap_api_key:
        return {"error": "未配置AMAP_API_KEY环境变量，无法搜索地点"}

    if center_lng is not None and center_lat is not None:
        url = settings.amap_place_around_url
        params = {
            "keywords": keyword,
            "location": f"{center_lng},{center_lat}",
            "radius": radius,
            "offset": min(top_n, 25),
            "key": settings.amap_api_key,
        }
    else:
        url = settings.amap_place_text_url
        params = {
            "keywords": keyword,
            "city": "武汉",
            "citylimit": "true",
            "offset": min(top_n, 25),
            "key": settings.amap_api_key,
        }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": f"地点搜索失败：{e}"}

    if data.get("status") != "1":
        return {"error": f"高德接口返回异常：{data.get('info')}"}

    pois = data.get("pois") or []
    results = [_parse_poi(p) for p in pois[:top_n] if p.get("location")]
    if not results:
        return {"error": f"没搜到「{keyword}」相关的地点，换个关键词试试"}
    return {"count": len(results), "results": results}

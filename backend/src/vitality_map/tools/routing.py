# ==============================================================
#  路线规划：真实路线查询（高德Direction API）+ 多点访问顺序排序（本地计算）
# ==============================================================

import itertools
import math

import requests

from vitality_map.core.config import AMAP_DIRECTION_URLS, settings


def _parse_amap_polyline(polyline_str: str) -> list[list[float]]:
    """高德polyline字符串"lng,lat;lng,lat;..."解析成[[lng,lat],...]坐标数组"""
    points = []
    for pair in polyline_str.split(";"):
        if not pair:
            continue
        lng_str, lat_str = pair.split(",")
        points.append([float(lng_str), float(lat_str)])
    return points


def _extract_name(field) -> str | None:
    """高德transit接口的entrance/exit字段文档写的是单个对象{"name":...}，但真实
    调用中(2026-08-20线上复现过一次崩溃)有时会返回一个列表——本地复现确认过
    seg["exit"]是list时对它调.get()直接AttributeError，把整条SSE流冲垮(模式B
    没有像模式A那样给每个工具调用包try/except，异常会一路网上传，见
    orchestrator/tool_wrap.py的修复)。这里防御式地兼容dict/list两种真实出现过
    的形状，都取不到就返回None。"""
    if isinstance(field, dict):
        return field.get("name")
    if isinstance(field, list) and field and isinstance(field[0], dict):
        return field[0].get("name")
    return None


def tool_route_between(origin_lng: float, origin_lat: float, dest_lng: float, dest_lat: float,
                        mode: str = "driving") -> dict:
    """
    两点间真实路线，接高德Direction API。mode: driving(驾车)/walking(步行)/
    transit(公交+地铁，含换乘站和地铁出入口信息)。返回给模型的字段只有精简摘要
    （距离/耗时/公交地铁线路名+上下车站+出入口）；真实道路坐标串存在"_polyline"
    这个下划线开头的字段里——agent喂给模型前会把下划线字段过滤掉（模型不需要
    几百个坐标点，那样只会浪费token），但会保留给前端画在地图上。
    """
    if not settings.amap_api_key:
        return {"error": "未配置AMAP_API_KEY环境变量，无法查询路线"}
    url = AMAP_DIRECTION_URLS.get(mode)
    if not url:
        return {"error": f"不支持的出行方式：{mode}，可选driving/walking/transit"}

    params = {
        "origin": f"{origin_lng},{origin_lat}",
        "destination": f"{dest_lng},{dest_lat}",
        "key": settings.amap_api_key,
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
            entrance_name = _extract_name(seg.get("entrance"))
            if entrance_name:
                entry["entrance"] = entrance_name
            exit_name = _extract_name(seg.get("exit"))
            if exit_name:
                entry["exit"] = exit_name
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


def tool_plan_route_order(points: list[dict]) -> dict:
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

# ==============================================================
#  从子agent内部的工具执行结果里提取地图元素，跟api/chat.py里模式A用的
#  _extract_map_features是同一份逻辑、同一套字段约定（markers/polylines/
#  seen_grid_ids），只是搬到这里给模式B的两个子agent复用。多了search_poi
#  的分支——search_poi一次会返回多个候选点，每个都产出一个marker。
#
#  跟模式A的原则保持一致：这里只负责"收集这一路上出现过的候选"，不代表这些
#  东西一定会显示在地图上——真正决定highlight_grid_ids的是Orchestrator的
#  finish工具，从seen_grid_ids里白名单校验+主动挑选（见orchestrator/finish.py）。
#  markers/polylines(geocode/search_poi/route_between产出)则是"查到即展示"，
#  跟格网高亮的语义不同，这点也跟模式A完全一致。
# ==============================================================


def extract_map_features(tool_name: str, result: dict) -> dict:
    if tool_name == "geocode" and "lng" in result and "lat" in result:
        return {"markers": [{"name": result.get("name"), "lng": result["lng"], "lat": result["lat"]}]}
    if tool_name == "search_poi" and result.get("results"):
        return {
            "markers": [
                {"name": r["name"], "lng": r["lng"], "lat": r["lat"]}
                for r in result["results"]
                if r.get("lng") is not None and r.get("lat") is not None
            ]
        }
    if tool_name == "route_between" and result.get("_polyline"):
        return {"polylines": [result["_polyline"]]}
    if tool_name == "get_vitality":
        return {"seen_grid_ids": [r["grid_id"] for r in result.get("results", [])]}
    if tool_name == "search_weibo_hotspots":
        return {"seen_grid_ids": [p["grid_id"] for p in result.get("posts", [])]}
    return {}

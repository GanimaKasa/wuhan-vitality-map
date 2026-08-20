# ==============================================================
#  路线子agent的工具集：geocode/search_poi/plan_route_order/route_between。
#  跟info_tools.py同样的薄包装模式，见tool_wrap.py。
# ==============================================================

from langchain.tools import ToolRuntime, tool

from vitality_map.agents.orchestrator.tool_wrap import build_command, emit_tool_call, safe_call
from vitality_map.tools.geocode import tool_geocode, tool_search_poi
from vitality_map.tools.routing import tool_plan_route_order, tool_route_between


@tool
def geocode(address: str, runtime: ToolRuntime):
    """把一个地名转成精确经纬度坐标（武汉范围内），地名不够精确时查不到会报错，
    换个更常见的说法再试。"""
    emit_tool_call("geocode", {"address": address})
    result = safe_call(tool_geocode, address=address)
    return build_command("geocode", result, runtime.tool_call_id)


@tool
def search_poi(keyword: str, runtime: ToolRuntime, center_lng: float | None = None,
                center_lat: float | None = None, radius: int = 3000, top_n: int = 10):
    """按关键字搜索地点(POI)，比web_search更精确、结构化。传center_lng/center_lat
    时在该点周边radius米内搜(比如"黄鹤楼附近的餐厅")，不传则在武汉全城范围搜。"""
    args = {"keyword": keyword, "center_lng": center_lng, "center_lat": center_lat,
            "radius": radius, "top_n": top_n}
    emit_tool_call("search_poi", args)
    result = safe_call(tool_search_poi, keyword=keyword, center_lng=center_lng, center_lat=center_lat,
                        radius=radius, top_n=top_n)
    return build_command("search_poi", result, runtime.tool_call_id)


@tool
def plan_route_order(points: list[dict], runtime: ToolRuntime):
    """给多个候选打卡点排一个访问顺序（按直线距离最优排序，不是精确路网距离）。
    points每项要有name/lng/lat三个字段。"""
    emit_tool_call("plan_route_order", {"points": points})
    result = safe_call(tool_plan_route_order, points=points)
    return build_command("plan_route_order", result, runtime.tool_call_id)


@tool
def route_between(origin_lng: float, origin_lat: float, dest_lng: float, dest_lat: float,
                   runtime: ToolRuntime, mode: str = "driving"):
    """查两点间的真实路线（driving驾车/walking步行/transit公交地铁，transit含地铁
    换乘站和出入口信息）。按plan_route_order排好的顺序，对相邻两点逐段调用。"""
    args = {"origin_lng": origin_lng, "origin_lat": origin_lat, "dest_lng": dest_lng,
            "dest_lat": dest_lat, "mode": mode}
    emit_tool_call("route_between", args)
    result = safe_call(tool_route_between, origin_lng=origin_lng, origin_lat=origin_lat,
                        dest_lng=dest_lng, dest_lat=dest_lat, mode=mode)
    return build_command("route_between", result, runtime.tool_call_id)


ROUTE_TOOLS = [geocode, search_poi, plan_route_order, route_between]

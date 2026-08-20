# ==============================================================
#  信息查询子agent的工具集：is_workday/get_weather/get_vitality/
#  search_weibo_hotspots/web_search。每个都是对tools/包里现成tool_*函数的
#  薄包装——真正的业务逻辑不重写，只是把返回值打包成Command(见tool_wrap.py)。
# ==============================================================

from langchain.tools import ToolRuntime, tool

from vitality_map.agents.orchestrator.tool_wrap import build_command, emit_tool_call
from vitality_map.tools.calendar import tool_is_workday
from vitality_map.tools.vitality import tool_get_vitality
from vitality_map.tools.weather import tool_get_weather
from vitality_map.tools.weibo import tool_search_weibo_hotspots
from vitality_map.tools.web_search import tool_web_search


@tool
def is_workday(date: str, runtime: ToolRuntime):
    """判断某天是工作日还是休息日，含中国法定节假日调休规则。date为YYYY-MM-DD格式。"""
    emit_tool_call("is_workday", {"date": date})
    return build_command("is_workday", tool_is_workday(date=date), runtime.tool_call_id)


@tool
def get_weather(date: str, runtime: ToolRuntime):
    """查询武汉某天的天气预报，仅支持未来3天（含今天），超出范围会返回错误——
    这时应该改用web_search查中期预报，不要直接放弃。date为YYYY-MM-DD格式。"""
    emit_tool_call("get_weather", {"date": date})
    return build_command("get_weather", tool_get_weather(date=date), runtime.tool_call_id)


@tool
def get_vitality(runtime: ToolRuntime, period: str | None = None, district: str | None = None,
                  order: str = "desc", topn: int = 10):
    """查询武汉三环内某时段/行政区的城市活力预测排名（基于多模态深度学习模型）。
    period如"工作日_日间"，不传则用综合平均；order是desc(高到低)或asc(低到高)。"""
    args = {"period": period, "district": district, "order": order, "topn": topn}
    emit_tool_call("get_vitality", args)
    result = tool_get_vitality(period=period, district=district, order=order, topn=topn)
    return build_command("get_vitality", result, runtime.tool_call_id)


@tool
def search_weibo_hotspots(keyword: str, runtime: ToolRuntime, top_n: int = 20):
    """语义检索微博热点，查真实脱敏微博用户在讨论/去哪些地方，反映本地人的真实动态。"""
    emit_tool_call("search_weibo_hotspots", {"keyword": keyword, "top_n": top_n})
    result = tool_search_weibo_hotspots(keyword=keyword, top_n=top_n)
    return build_command("search_weibo_hotspots", result, runtime.tool_call_id)


@tool
def web_search(query: str, runtime: ToolRuntime):
    """通用网页搜索，用于微博数据/城市活力数据覆盖不到的开放性信息，结果可能不够
    准确，回答时要提醒用户"仅供参考"。"""
    emit_tool_call("web_search", {"query": query})
    return build_command("web_search", tool_web_search(query=query), runtime.tool_call_id)


INFO_TOOLS = [is_workday, get_weather, get_vitality, search_weibo_hotspots, web_search]

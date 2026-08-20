# ==============================================================
#  子agent的每个工具执行完之后，统一用这个函数打包成Command双通道回传：
#    - messages：结果(过滤掉下划线开头字段，跟模式A同样的"不喂坐标串给模型"
#      原则)序列化成JSON字符串，包成ToolMessage给模型看
#    - markers/polylines/seen_grid_ids：从map_features.extract_map_features
#      提取，直接写进子agent自己的state，不经过LLM
#
#  这套双通道机制是LangChain/LangGraph官方文档"Tools -> Update state"一节
#  明确给出的写法(Command(update={...}))，真实调用验证过。每个工具在
#  info_tools.py/route_tools.py里显式定义(不用动态拼签名的工厂函数)——
#  9个工具的规模，显式写比自省函数签名这种取巧写法更好维护、更不容易踩坑。
# ==============================================================

import json

from langchain.messages import ToolMessage
from langgraph.config import get_stream_writer
from langgraph.types import Command

from vitality_map.agents.orchestrator.map_features import extract_map_features
from vitality_map.agents.single_agent import TOOL_DISPLAY_NAMES


def _visible(result: dict) -> dict:
    """下划线开头的字段(如route_between的_polyline)只给地图用，不喂给模型，
    跟模式A run_agent_stream()里的model_visible_result是同一个过滤规则。"""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def safe_call(impl, **kwargs) -> dict:
    """跟模式A(single_agent.run_agent_stream的try/except impl(**args))同样的
    容错：真实生产事故(2026-08-20)确认过，模式B原来直接调impl(**kwargs)，一旦
    工具内部抛异常(这次是tools/routing.py的一个真实bug)，异常会一路网上传穿过
    子agent→委派工具→Orchestrator，把整条SSE流冲垮——FastAPI的StreamingResponse
    生成器一旦抛异常就静默关闭连接，前端永远收不到final事件，卡在最后一步的
    "正在..."状态出不来。现在统一在这里兜底，工具异常转成{"error":...}结果，
    模型能看到报错、可以换个思路继续，不会让整轮对话彻底死掉。"""
    try:
        return impl(**kwargs)
    except Exception as e:
        return {"error": str(e)}


def emit_tool_call(tool_name: str, args: dict) -> None:
    """执行前发一个"开始调用"信号，用get_stream_writer()——真实调用验证过：
    直接写在子agent自己的StateGraph执行上下文里能被subagent.stream()正常收到，
    但不会自动冒泡到外层Orchestrator的stream()，需要delegate工具那一层手动转发
    （见subagents.py），这是LangGraph嵌套子图的真实行为，不是自动传播的。"""
    writer = get_stream_writer()
    label = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
    writer({"type": "tool_call", "tool": tool_name, "label": label, "args": args})


def build_command(tool_name: str, result: dict, tool_call_id: str, extra_update: dict | None = None) -> Command:
    """执行后打包结果：既发"调用结果"流式信号给前端展示，也用Command双通道
    把结果写回state（文字给模型看+markers/polylines/seen_grid_ids绕过LLM）。
    extra_update是额外要合并进state的字段(目前只有route_tools.py的geocode/
    search_poi用来写poi_cache，见state.py)。"""
    visible = _visible(result)
    label = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
    get_stream_writer()({"type": "tool_result", "tool": tool_name, "label": label, "result": visible})

    features = extract_map_features(tool_name, result) or {}
    update = {
        "markers": features.get("markers", []),
        "polylines": features.get("polylines", []),
        "seen_grid_ids": features.get("seen_grid_ids", []),
        "messages": [
            ToolMessage(content=json.dumps(visible, ensure_ascii=False), tool_call_id=tool_call_id)
        ],
    }
    if extra_update:
        update.update(extra_update)
    return Command(update=update)

# ==============================================================
#  统一问答入口："问答"和"智能推荐"原来是两个独立入口，合并成一个/api/chat：
#  parse_intent能命中时（时段/行政区/高低方向都是关键词能可靠解析的封闭式
#  问题）走快速路径（单次DeepSeek调用），命不中、或问题带开放性信号词时才走
#  agent循环（慢/贵，但SSE流式返回让用户能看到中间步骤）。
# ==============================================================

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from vitality_map.agents import single_agent
from vitality_map.agents.orchestrator import resume_orchestrator_stream, run_orchestrator_stream
from vitality_map.core.rate_limit import check_rate_limit
from vitality_map.retrieval import intent as retrieval
from vitality_map.core.data import KNOWN_DISTRICTS
from vitality_map.schemas.chat import ChatRequest
from vitality_map.services import llm_client
from vitality_map.tools.calendar import tool_is_workday
from vitality_map.tools.geocode import tool_geocode
from vitality_map.tools.routing import tool_plan_route_order, tool_route_between
from vitality_map.tools.vitality import resolve_periods_and_rows, tool_get_vitality
from vitality_map.tools.weather import tool_get_weather
from vitality_map.tools.weibo import tool_search_weibo_hotspots
from vitality_map.tools.web_search import tool_web_search

router = APIRouter(tags=["chat"])

# 快速路径判断用的关键词表
AGENT_TRIGGER_KEYWORDS = ["推荐", "路线", "规划", "去哪", "怎么玩", "一日游", "行程", "打卡", "周末"]

# 快速路径是纯关键词解析+单次数据库查询，完全不知道"历史对话"这回事——如果问题里
# 恰好也命中了period/district/direction关键词（哪怕本意是追问上一轮，比如"你刚才说的
# 活力最高那个格网"里的"活力""高"会命中direction关键词），会被判定has_signal=True走
# 快速路径，安安静静地重新查一次全新数据，跟上一轮答案对不上却不会报错——这是真实
# 复现过的bug，不是假设：只要问题里带这类"指代之前对话"的词、且这次请求确实带了历史，
# 就强制走agent路径（有历史记忆能力），不看关键词解析结果如何。
MEMORY_REFERENCE_KEYWORDS = ["刚才", "刚刚", "刚说", "上面", "上一条", "上次", "之前", "你说", "提到的"]

# agent路径的finish由模型自己判断该不该高亮格网（见agents/single_agent.py），但模型的
# 判断会有偏差——真实复现过"推荐8个打卡点+规划路线"这类问题，8个高亮格网里有2个是
# 模型顺手从一次不相关的"全城活力排名"查询里带上的，跟推荐的具体地点没关系。这里加
# 一道硬性兜底：用户这轮问题本身（不是历史）压根没提"活力/热闹/冷清"这类词的话，直接
# 不展示格网高亮，不管模型选了什么——不是猜"有没有路线"这种间接信号，是直接检查最
# 相关的那个信号本身（用户是不是真的问了活力相关的问题），复用retrieval.intent里
# parse_direction已经在用的同一份关键词表，不重新造一份。
VITALITY_KEYWORDS = set(retrieval.HIGH_KEYWORDS) | set(retrieval.LOW_KEYWORDS)

AGENT_TOOL_IMPLS = {
    "is_workday": tool_is_workday,
    "get_weather": tool_get_weather,
    "get_vitality": tool_get_vitality,
    "search_weibo_hotspots": tool_search_weibo_hotspots,
    "web_search": tool_web_search,
    "geocode": tool_geocode,
    "route_between": tool_route_between,
    "plan_route_order": tool_plan_route_order,
}


def _should_use_agent(question: str, intent: dict, has_history: bool = False) -> bool:
    has_signal = bool(intent["periods"] or intent["district"] or intent["direction"])
    has_open_ended_kw = any(kw in question for kw in AGENT_TRIGGER_KEYWORDS)
    references_memory = has_history and any(kw in question for kw in MEMORY_REFERENCE_KEYWORDS)
    return has_open_ended_kw or not has_signal or references_memory


def _suppress_ungrounded_highlight(event: dict, question: str) -> dict:
    if event.get("type") != "final" or not event.get("highlight_grid_ids"):
        return event
    if any(kw in question for kw in VITALITY_KEYWORDS):
        return event
    return {**event, "highlight_grid_ids": []}


def _extract_map_features(tool_name: str, result: dict) -> dict:
    """
    从工具结果里挑出能在地图上画出来的东西：
    - markers：geocode查到的地点，前端画点用
    - polylines：route_between查到的真实道路坐标串（存在"_polyline"字段里，
      不会被喂给模型，只在这里、只给地图用）
    - seen_grid_ids：get_vitality/search_weibo_hotspots这次结果里出现过的格网编号。
      **不会自动展示在地图上**——格网高亮只来自模型显式调用finish(answer,
      highlight_grid_ids)时主动挑选的，见agents/single_agent.py里的说明。
      seen_grid_ids只是用来做白名单校验的事实依据：模型在finish里选的
      highlight_grid_ids必须是这一路真实查询到过的格网子集，防止编造/记错一个
      从没查到过的编号却被原样显示到地图上。
    """
    if tool_name == "geocode" and "lng" in result and "lat" in result:
        return {"markers": [{"name": result.get("name"), "lng": result["lng"], "lat": result["lat"]}]}
    if tool_name == "route_between" and result.get("_polyline"):
        return {"polylines": [result["_polyline"]]}
    if tool_name == "get_vitality":
        return {"seen_grid_ids": [r["grid_id"] for r in result.get("results", [])]}
    if tool_name == "search_weibo_hotspots":
        return {"seen_grid_ids": [p["grid_id"] for p in result.get("posts", [])]}
    return {}


@router.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    """
    统一问答入口，SSE流式返回。快速路径只发一个"final"事件（跟agent路径的最终
    事件同格式），前端不用区分走的是哪条路径，都当成同一套事件流处理；agent路径
    会先发若干"tool_call"/"tool_result"事件展示中间步骤，中途可能发一个"ask_user"
    事件暂停等用户回答，最终发"final"事件。

    pending_turn有值时，说明这是在恢复一次被ask_user打断的对话——直接进agent循环
    续接，不再走parse_intent那套快速路径判断（快速路径设计上就是处理独立的、封闭式
    的单轮问题，跟"接着上次没问完的话题继续"这件事没关系）。
    """
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    def event_stream():
        if req.agent_mode == "orchestrator":
            # 模式B(LangGraph multi-agent)。前端还没有切换按钮，目前只能手动在
            # 请求体里传agent_mode="orchestrator"测试。ask_user中途反问已经接了
            # checkpointer(见agents/orchestrator/checkpointer.py)，pending_turn
            # 这里只需要带thread_id——跟模式A(pending_turn带完整messages快照)
            # 结构不同，是两套并存的机制，api层只是原样透传，不需要互相兼容。
            if req.pending_turn:
                thread_id = req.pending_turn.get("thread_id")
                original_question = req.question or ""
                for event in resume_orchestrator_stream(thread_id, req.reply or ""):
                    event = _suppress_ungrounded_highlight(event, original_question)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                return

            question = req.question or ""
            for event in run_orchestrator_stream(question, history=req.history):
                event = _suppress_ungrounded_highlight(event, question)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            return

        if req.pending_turn:
            # req.question在恢复场景下是前端透传回来的"这一整轮最初的问题"（不是这次
            # 回复的文字），用来做下面的活力关键词兜底检查——跟全新一轮走同一套判断。
            original_question = req.question or ""
            for event in single_agent.run_agent_stream(
                AGENT_TOOL_IMPLS, _extract_map_features,
                pending_turn=req.pending_turn, reply=req.reply,
            ):
                event = _suppress_ungrounded_highlight(event, original_question)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            return

        question = req.question or ""
        parsed_intent = retrieval.parse_intent(question, KNOWN_DISTRICTS)
        use_agent = _should_use_agent(question, parsed_intent, bool(req.history))

        if use_agent:
            for event in single_agent.run_agent_stream(
                AGENT_TOOL_IMPLS, _extract_map_features,
                question=question, history=req.history,
            ):
                event = _suppress_ungrounded_highlight(event, question)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            return

        has_signal = bool(parsed_intent["periods"] or parsed_intent["district"] or parsed_intent["direction"])
        if has_signal:
            rows = resolve_periods_and_rows(parsed_intent)
            context = {"intent": parsed_intent, "rows": rows}
            answer = llm_client.answer_question(question, context)
            highlight_grid_ids = [r["grid_id"] for r in rows]
        else:
            answer = llm_client.answer_question(question, {"fallback": True})
            highlight_grid_ids = []
        event = {"type": "final", "answer": answer, "highlight_grid_ids": highlight_grid_ids}
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

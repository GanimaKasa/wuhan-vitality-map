# ==============================================================
#  模式B(multi-agent, LangGraph)对外入口。
#
#  run_orchestrator_stream()是SSE流式版本，事件格式跟模式A(single_agent.
#  run_agent_stream)对齐——{"type":"tool_call",...}/{"type":"tool_result",...}/
#  {"type":"ask_user",...}/{"type":"final",...}，前端不用区分走的是模式A还是
#  模式B。ask_user中途反问现在接了真checkpointer(见checkpointer.py)，
#  pending_turn只需要带thread_id(不像模式A要带完整messages快照——状态已经
#  在服务器端持久化了，这是接checkpointer换来的简化)。
#
#  用stream_mode=["custom","values"]：custom收集子agent内部工具emit的
#  tool_call/tool_result信号(经由subagents.py._run_subagent_and_relay手动
#  转发上来的，注意事项见项目记忆——custom不会自动穿透嵌套子图)，values收集
#  每一步之后的完整state快照，用最后一次values判断是"正常结束"还是"被
#  ask_user暂停"(state里出现"__interrupt__"键，真实调用验证过的判断依据)。
# ==============================================================

import uuid

from langchain.messages import HumanMessage
from langgraph.types import Command

from vitality_map.agents.orchestrator.graph import build_orchestrator


def _build_messages(question: str, history: list[dict] | None):
    messages = []
    for turn in (history or []):
        q, a = turn.get("question"), turn.get("answer")
        if q and a:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
    messages.append(HumanMessage(content=question))
    return messages


def _drive_stream(orchestrator, thread_id: str, input_):
    config = {"configurable": {"thread_id": thread_id}}
    final_state = None
    for kind, data in orchestrator.stream(input_, config=config, stream_mode=["custom", "values"]):
        if kind == "custom":
            yield data
        elif kind == "values":
            final_state = data

    if final_state is None:
        yield {"type": "final", "answer": "抱歉，推荐服务暂时不可用，请稍后再试。",
               "highlight_grid_ids": [], "markers": [], "polylines": []}
        return

    interrupts = final_state.get("__interrupt__")
    if interrupts:
        payload = interrupts[0].value
        yield {
            "type": "ask_user",
            "question": payload.get("question") or "能再说得具体一点吗？",
            "options": payload.get("options"),
            "pending_turn": {"thread_id": thread_id},
        }
        return

    yield {
        "type": "final",
        "answer": final_state["messages"][-1].content,
        "highlight_grid_ids": final_state.get("highlight_grid_ids", []),
        "markers": final_state.get("markers", []),
        "polylines": final_state.get("polylines", []),
    }


def run_orchestrator(question: str, history: list[dict] | None = None) -> dict:
    """非流式版本，测试/脚本场景用。返回格式跟run_orchestrator_stream()的
    最终final事件一致（不支持ask_user暂停，遇到会把中断payload原样塞进answer，
    只用于快速验证，正式走流式接口）。"""
    orchestrator = build_orchestrator()
    thread_id = str(uuid.uuid4())
    events = list(_drive_stream(orchestrator, thread_id,
                                 {"messages": _build_messages(question, history)}))
    return events[-1]


def run_orchestrator_stream(question: str, history: list[dict] | None = None):
    """全新一轮（或者上一轮已正常结束、紧接着问的新问题），跟模式A的
    run_agent_stream(question=..., history=...)对应。"""
    orchestrator = build_orchestrator()
    thread_id = str(uuid.uuid4())
    yield from _drive_stream(orchestrator, thread_id, {"messages": _build_messages(question, history)})


def resume_orchestrator_stream(thread_id: str, reply: str):
    """上一轮被ask_user打断、这次是恢复，跟模式A的run_agent_stream(pending_turn=...,
    reply=...)对应。thread_id来自上次ask_user事件里的pending_turn，原样带回来。"""
    orchestrator = build_orchestrator()
    yield from _drive_stream(orchestrator, thread_id, Command(resume=reply))

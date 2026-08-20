# ==============================================================
#  构建两个子agent(create_agent实例)，并把它们各自包装成Orchestrator能调用的
#  "委派"工具——这是LangChain官方"Subagents"模式的"Tool per agent"写法
#  (main agent把subagent当工具调用)，真实调用验证过，见项目记忆。
#
#  子agent内部执行时，它自己的工具(info_tools.py/route_tools.py里定义的)会用
#  Command把markers/polylines/seen_grid_ids写进*子agent自己的*state——子agent
#  跑完之后，这里的委派工具再从子agent返回的最终state里读出这些字段，通过自己
#  的Command update写进*Orchestrator的*state，实现"结果两条通道往上冒泡"：
#  文字总结(result["messages"][-1].content)进Orchestrator的消息历史给它看，
#  markers/polylines/seen_grid_ids绕过LLM直接进Orchestrator state。
#
#  子agent只看得到Orchestrator委派的这一句task文本(HumanMessage)，看不到
#  Orchestrator和用户的完整历史——这就是"子agent上下文隔离"，也是这次做
#  multi-agent重构相对模式A(单体)的核心提升点之一(见项目记忆里"agent上下文
#  管理要升级的具体差距")。
# ==============================================================

from langchain.agents import create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.config import get_stream_writer
from langgraph.types import Command

from vitality_map.agents.orchestrator.checkpointer import get_checkpointer
from vitality_map.agents.orchestrator.info_tools import INFO_TOOLS
from vitality_map.agents.orchestrator.llm import get_deepseek_chat_model
from vitality_map.agents.orchestrator.prompts import INFO_AGENT_SYSTEM_PROMPT, ROUTE_AGENT_SYSTEM_PROMPT
from vitality_map.agents.orchestrator.route_tools import ROUTE_TOOLS
from vitality_map.agents.orchestrator.state import OrchestratorState


def _build_subagent(system_prompt: str, tools: list):
    # 子agent也接同一个checkpointer单例——2026-08-20真实测试验证过：如果不接，
    # Orchestrator进程崩溃在"子agent执行到一半"这个时间点，子agent自己已经跑完
    # 的那些工具调用完全没有留痕，恢复后delegate工具会被Orchestrator的checkpoint
    # 重放，子agent从零开始重跑全部步骤——现有工具都是只读查询，重跑不会导致
    # 数据错乱，但"精准恢复不重复劳动"这层保证是缺失的，是真实验证出来的bug。
    return create_agent(
        model=get_deepseek_chat_model(),
        tools=tools,
        system_prompt=system_prompt,
        state_schema=OrchestratorState,
        checkpointer=get_checkpointer(),
    )


def _bubble_up(subagent_final_state: dict, tool_call_id: str) -> Command:
    summary = subagent_final_state["messages"][-1].content
    return Command(update={
        "markers": subagent_final_state.get("markers", []),
        "polylines": subagent_final_state.get("polylines", []),
        "seen_grid_ids": subagent_final_state.get("seen_grid_ids", []),
        "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
    })


def _run_subagent_and_relay(subagent, task: str, sub_thread_id: str) -> dict:
    """跑子agent，用stream(stream_mode=["custom","values"])而不是invoke()——
    自定义流式信号(子agent内部工具emit_tool_call/build_command发的
    tool_call/tool_result)不会自动冒泡穿过嵌套的子图调用(真实测试验证过，
    见项目记忆)，必须在这一层手动转发给外层Orchestrator的writer，前端才能
    实时看到子agent内部到底在做什么，不是只看到一个"委派中..."的黑箱。

    sub_thread_id是从"父thread_id+这次委派调用的tool_call_id"派生出来的稳定
    标识——如果Orchestrator崩溃后从checkpoint重放同一次delegate工具调用，
    tool_call_id是不变的(同一条AIMessage里的同一次tool_call)，派生出的
    sub_thread_id也就跟上次崩溃前是同一个，子agent的checkpointer能认出这是
    "接着上次的进度跑"而不是全新任务。先查一次这个thread_id有没有已存在的
    state，有就传None表示"从checkpoint继续"，没有就传全新的任务消息。"""
    config = {"configurable": {"thread_id": sub_thread_id}}
    existing = subagent.get_state(config)
    input_ = None if existing.values.get("messages") else {"messages": [HumanMessage(content=task)]}

    outer_writer = get_stream_writer()
    final_state = None
    for kind, data in subagent.stream(input_, config=config, stream_mode=["custom", "values"]):
        if kind == "custom":
            outer_writer(data)
        elif kind == "values":
            final_state = data
    return final_state


_info_subagent = None
_route_subagent = None


def _get_info_subagent():
    global _info_subagent
    if _info_subagent is None:
        _info_subagent = _build_subagent(INFO_AGENT_SYSTEM_PROMPT, INFO_TOOLS)
    return _info_subagent


def _get_route_subagent():
    global _route_subagent
    if _route_subagent is None:
        _route_subagent = _build_subagent(ROUTE_AGENT_SYSTEM_PROMPT, ROUTE_TOOLS)
    return _route_subagent


def _derive_sub_thread_id(runtime: ToolRuntime) -> str:
    parent_thread_id = runtime.config["configurable"]["thread_id"]
    return f"{parent_thread_id}:{runtime.tool_call_id}"


@tool
def delegate_to_info_agent(task: str, runtime: ToolRuntime):
    """把日历/天气/城市活力/微博热点/开放性网页信息类任务委派给信息查询子agent。
    task要写清楚具体要查什么(含必要的日期/地点等上下文)，子agent看不到你和用户的
    完整对话历史。"""
    final_state = _run_subagent_and_relay(_get_info_subagent(), task, _derive_sub_thread_id(runtime))
    return _bubble_up(final_state, runtime.tool_call_id)


@tool
def delegate_to_route_agent(task: str, runtime: ToolRuntime):
    """把地点查询/打卡点发现/路线规划类任务委派给路线子agent。task要写清楚具体
    要查什么(含必要的地名/出行方式等上下文)，子agent看不到你和用户的完整对话历史。"""
    final_state = _run_subagent_and_relay(_get_route_subagent(), task, _derive_sub_thread_id(runtime))
    return _bubble_up(final_state, runtime.tool_call_id)

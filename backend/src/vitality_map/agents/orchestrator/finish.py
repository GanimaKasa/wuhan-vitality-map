# ==============================================================
#  finish工具：Orchestrator用它结束对话、给出最终回答，跟模式A(single_agent.py)
#  里的finish是同一个设计——return_direct=True让调用完这个工具后agent循环立刻
#  结束，不再多绕一轮回模型(LangChain官方"Return directly from a tool"文档写明
#  的机制)，highlight_grid_ids做白名单校验(runtime.state读seen_grid_ids，
#  只保留模型这一路真实委派子agent查询到过的编号，防止编造/记错编号)。
# ==============================================================

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command


@tool(return_direct=True)
def finish(answer: str, runtime: ToolRuntime, highlight_grid_ids: list[int] | None = None) -> Command:
    """结束对话并给出最终回答。综合两个子agent汇报的信息、想清楚要怎么回答用户后，
    必须调用这个工具，不要不调用任何工具就直接用文字回复结束对话。
    answer是给用户看的最终回答，口语化、有理有据。highlight_grid_ids是要在地图上
    高亮的格网编号，从子agent汇报过程中出现过的候选格网里主动挑选真正构成这轮推荐
    依据的那些；不需要强调任何格网时传空数组，不要为了有内容而塞进无关格网。"""
    seen = set(runtime.state.get("seen_grid_ids") or [])
    raw = highlight_grid_ids or []
    whitelisted = [g for g in raw if isinstance(g, int) and g in seen]
    return Command(update={
        "highlight_grid_ids": whitelisted,
        "messages": [ToolMessage(content=answer, tool_call_id=runtime.tool_call_id)],
    })

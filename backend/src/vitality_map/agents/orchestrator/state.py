# ==============================================================
#  模式B(multi-agent)的Orchestrator图状态。
#
#  markers/polylines/seen_grid_ids这三个字段是"结构化侧信道"——不给模型看，
#  只用来在子agent/工具执行时累积地图数据，最终由api层读出来发给前端。子agent
#  通过Command(update=...)直接写这几个字段，不经过LLM消息通道，跟模式A单体版
#  run_agent_stream()里本地变量markers/polylines/seen_grid_ids是同一个角色，
#  只是从"函数局部变量"变成了"图状态字段"。
#
#  reducer说明：LangGraph在有多个并行工具调用同时更新同一个字段时，需要一个
#  reducer函数决定怎么合并。markers/polylines用简单列表拼接(+)；seen_grid_ids
#  用去重union，因为get_vitality/search_weibo_hotspots可能在同一批并行调用里
#  都命中同一个格网编号，不去重的话白名单校验时会有大量重复但不影响正确性——
#  去重只是让state更干净。
# ==============================================================

from typing import Annotated

from langchain.agents import AgentState


def _union_grid_ids(a: list[int] | None, b: list[int] | None) -> list[int]:
    return sorted(set(a or []) | set(b or []))


class OrchestratorState(AgentState):
    today: str
    markers: Annotated[list[dict], lambda a, b: (a or []) + (b or [])]
    polylines: Annotated[list[list], lambda a, b: (a or []) + (b or [])]
    seen_grid_ids: Annotated[list[int], _union_grid_ids]
    # finish工具(return_direct=True)写入，api层读出来作为这一轮最终结果
    highlight_grid_ids: list[int]

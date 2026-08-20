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
    # "长期记忆"锚点：Orchestrator=用户最初的问题，子agent=Orchestrator委派的task
    # 原文。故意不放进messages里——SummarizationMiddleware(见graph.py/subagents.py)
    # 只会摘要messages，这个字段完全不受影响，靠prompts.py里的dynamic_prompt钩子
    # 每次模型调用前都显式塞回系统提示词。真实测试验证过(见项目记忆)：哪怕messages
    # 已经被摘要替换、原始问题的文字被压缩没了，模型依然能通过这个字段准确复述
    # 最初的任务目标，不会像"无差别历史截断"那样把任务目标弄丢。
    original_question: str
    # "跨委派POI缓存"：修2026-08-19用户实测反馈的"重复搜索同一地标"问题——
    # geocode/search_poi的结果按归一化key存这里，key -> 原始result。Orchestrator
    # 对路线子agent做多次独立委派时，每次都是全新子agent线程(为了上下文隔离特意
    # 这样设计的)，彼此不共享"已经查过什么"，真实复现过一次"推荐8个打卡点"里
    # 黄鹤楼/户部巷/昙华林等地标被重复搜索2~3遍。修复思路：委派前把Orchestrator
    # 这边已经攒下的缓存传给子agent当"已知候选"起点，子agent查询前先看缓存有没有
    # 命中，命中就跳过真正的高德API调用；子agent查完的新增缓存条目再合并回
    # Orchestrator这边，供下一次委派复用。见route_tools.py/subagents.py。
    poi_cache: Annotated[dict, lambda a, b: {**(a or {}), **(b or {})}]

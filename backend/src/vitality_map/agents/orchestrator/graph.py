# ==============================================================
#  组装Orchestrator：create_agent(委派工具x2 + finish) + 自定义state_schema。
#  跟模式A(single_agent.run_agent_stream)的关系见agents/single_agent.py顶部
#  注释——两者并存，对应前端的agent模式切换（切换按钮还没做，见项目路线图）。
#
#  system_prompt改成用orchestrator_prompt这个dynamic_prompt中间件，不是静态
#  字符串——每次模型调用前都重新拼一遍(含当天日期+"长期记忆"任务目标锚点)，
#  不会有"构建一次缓存住、日期冻结"的坑（这是模式A/single_agent.py注释里提到
#  过的已经线上踩过的bug同一类），所以这里也就不用再强调"故意不缓存单例"了。
#
#  SummarizationMiddleware是"中期记忆"：单轮对话内如果委派了很多轮子agent、
#  消息历史堆积过大，超过token阈值自动摘要旧消息、保留最近几条，官方内置
#  中间件，不用自己手搭状态机做这件事。trigger/keep数值凭经验给的初始值，
#  不是精确调优过的，如果实测发现太早/太晚触发可以调。
# ==============================================================

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from vitality_map.agents.orchestrator.ask_user import ask_user
from vitality_map.agents.orchestrator.checkpointer import get_checkpointer
from vitality_map.agents.orchestrator.finish import finish
from vitality_map.agents.orchestrator.llm import get_deepseek_chat_model
from vitality_map.agents.orchestrator.prompts import orchestrator_prompt
from vitality_map.agents.orchestrator.state import OrchestratorState
from vitality_map.agents.orchestrator.subagents import delegate_to_info_agent, delegate_to_route_agent


def build_orchestrator():
    return create_agent(
        model=get_deepseek_chat_model(),
        tools=[delegate_to_info_agent, delegate_to_route_agent, ask_user, finish],
        middleware=[
            SummarizationMiddleware(model=get_deepseek_chat_model(), trigger=("tokens", 6000),
                                     keep=("messages", 12)),
            orchestrator_prompt,
        ],
        state_schema=OrchestratorState,
        checkpointer=get_checkpointer(),
    )

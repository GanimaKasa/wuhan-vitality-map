# ==============================================================
#  组装Orchestrator：create_agent(委派工具x2 + finish) + 自定义state_schema。
#  跟模式A(single_agent.run_agent_stream)的关系见agents/single_agent.py顶部
#  注释——两者并存，对应前端的agent模式切换（切换按钮还没做，见项目路线图）。
#
#  故意不缓存单例：system_prompt里拼了当天日期(today_str())，如果构建一次就
#  缓存住，Render容器长期运行、进程不重启的情况下"今天"会永久冻结在进程启动
#  那一刻——这正是模式A(single_agent.py)注释里提到过的、已经线上踩过的时区/
#  日期bug的同一类坑，这里直接从设计上避免，代价是每次请求都重新构建一次graph
#  定义(轻量，不涉及网络调用，真正的开销在.invoke()时才发生)。
# ==============================================================

from langchain.agents import create_agent

from vitality_map.agents.orchestrator.ask_user import ask_user
from vitality_map.agents.orchestrator.checkpointer import get_checkpointer
from vitality_map.agents.orchestrator.finish import finish
from vitality_map.agents.orchestrator.llm import get_deepseek_chat_model
from vitality_map.agents.orchestrator.prompts import ORCHESTRATOR_SYSTEM_PROMPT_TEMPLATE, today_str
from vitality_map.agents.orchestrator.state import OrchestratorState
from vitality_map.agents.orchestrator.subagents import delegate_to_info_agent, delegate_to_route_agent


def build_orchestrator():
    return create_agent(
        model=get_deepseek_chat_model(),
        tools=[delegate_to_info_agent, delegate_to_route_agent, ask_user, finish],
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT_TEMPLATE.format(today=today_str()),
        state_schema=OrchestratorState,
        checkpointer=get_checkpointer(),
    )

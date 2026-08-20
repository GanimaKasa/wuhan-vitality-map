# ==============================================================
#  ask_user：只挂在Orchestrator上，不给两个子agent用——跟模式A(single_agent.py)
#  设计讨论时确认过的原则一致（见2026-08-20会话）：反问权限收归一处，避免
#  "子图内部又嵌套一层interrupt"的复杂状态机。子agent如果发现信息不够，
#  应该在汇报文字里说清楚"缺什么信息"，交由Orchestrator决定要不要真的暂停
#  问用户，而不是自己直接interrupt。
#
#  真实调用验证过（见项目记忆）：interrupt(payload)暂停执行，暂停状态由
#  checkpointer.py的SqliteSaver持久化；下次带着同一个thread_id + Command(
#  resume=answer)调用，会从暂停的地方原样接着跑，回答会作为这次interrupt()
#  调用的返回值(answer变量)交回ask_user函数、包成工具结果继续对话。
# ==============================================================

from langchain.tools import tool
from langgraph.types import interrupt


@tool
def ask_user(question: str, options: list[str] | None = None) -> str:
    """拿不准该怎么理解用户意图、或者某个决策最好让用户自己选时，调用这个工具
    主动停下来问用户一句，而不是自己瞎猜。调用后对话会暂停，等用户回答才继续，
    所以只在真的需要用户输入才能继续时才用，能自己合理判断的不要问。
    有明确的几个选项可选时，把它们列进options，用户会看到按钮直接点选；开放式
    问题（比如具体地名、数字，没法枚举）不传这个字段，用户会自己打字回答。"""
    payload: dict = {"question": question}
    if options:
        payload["options"] = options
    answer = interrupt(payload)
    return answer

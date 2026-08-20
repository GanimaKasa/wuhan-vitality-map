from pydantic import BaseModel


class ChatRequest(BaseModel):
    # question+history：全新一轮（或者上一轮已正常结束、紧接着问的新问题）。
    # history是前端本地压缩存好的最近几轮{question,answer}，不传等价于没有记忆。
    # pending_turn+reply：上一轮被ask_user打断、这次是恢复——pending_turn是上次
    # "ask_user"事件里原样发给前端、又原样传回来的快照，不做任何加工，直接透传给
    # agents.single_agent.run_agent_stream。两种方式二选一，都不传时question按空
    # 字符串处理。
    question: str | None = None
    history: list[dict] | None = None
    pending_turn: dict | None = None
    reply: str | None = None
    # "single"(默认，模式A单体ReAct agent) | "orchestrator"(模式B，LangGraph
    # multi-agent)。前端切换按钮还没做，目前只能手动传这个字段测试模式B——见
    # agents/orchestrator/包。orchestrator模式的pending_turn结构跟模式A不同：
    # 只需要{"thread_id": str}（真checkpointer持久化了完整状态，不需要像模式A
    # 那样客户端带回完整messages快照）。
    agent_mode: str = "single"

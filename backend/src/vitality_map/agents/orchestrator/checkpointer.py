# ==============================================================
#  ask_user中途反问的暂停/恢复(LangGraph原生interrupt()/Command(resume=...)
#  机制)必须依赖一个真正的checkpointer——不是长期存档，只是让"同一轮对话内
#  两次HTTP请求之间"这段时间的执行状态能被找回来。惰性单例：SqliteSaver
#  内部拿着一个sqlite3.Connection常驻，FastAPI进程存活期间复用，不用每次
#  请求重新开连接。
#
#  已知限制：Render免费档容器磁盘不持久化，重启后这个SQLite文件会被清空——
#  如果用户在"被反问"和"回答"之间恰好撞上容器重启/重新部署，这次暂停的对话
#  会找不到状态、恢复失败。这跟模式A的pending_turn(客户端原样带回完整快照，
#  完全不依赖服务器状态)相比是真实的可靠性代价，属于选择LangGraph原生机制
#  换来的权衡(见2026-08-20会话里的路线讨论)，不是疏漏。真要在Render上线用
#  这个功能，需要换成PostgresSaver+真Postgres实例。
# ==============================================================

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from vitality_map.core.config import settings

_saver: SqliteSaver | None = None


def get_checkpointer() -> SqliteSaver:
    global _saver
    if _saver is None:
        conn = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
        _saver = SqliteSaver(conn)
        _saver.setup()
    return _saver

# ==============================================================
#  简单内存限流：每IP每窗口期最多N次，防止公网调用真实LLM API被刷爆费用。
#  进程重启会清零；如果部署在反向代理后面，需要改成读X-Forwarded-For。
#  这是"目录重构"阶段的最小占位——真正能扛横向扩展的限流（比如接Redis）
#  属于第四阶段"后端工程化"要做的事。
# ==============================================================

import time
from collections import defaultdict

from fastapi import HTTPException

from vitality_map.core.config import settings

_request_log: dict[str, list[float]] = defaultdict(list)  # ip -> 请求时间戳列表


def check_rate_limit(ip: str) -> None:
    now = time.time()
    window_start = now - settings.chat_rate_window_seconds
    timestamps = _request_log[ip]
    while timestamps and timestamps[0] < window_start:
        timestamps.pop(0)
    if len(timestamps) >= settings.chat_rate_limit:
        raise HTTPException(
            status_code=429,
            detail=(
                f"请求太频繁，每个IP每{settings.chat_rate_window_seconds}秒最多"
                f"{settings.chat_rate_limit}次提问，请稍后再试。"
            ),
        )
    timestamps.append(now)

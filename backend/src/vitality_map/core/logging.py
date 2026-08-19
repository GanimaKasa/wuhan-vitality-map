# ==============================================================
#  基础日志配置：取代原来散落各处的print(..., flush=True)。
#  这里只是"目录重构"阶段的最小占位——统一成标准logging模块、有模块名/级别，
#  真正的结构化trace/成本埋点是第一阶段"可观测性"要做的事，不在这次重构范围内。
# ==============================================================

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

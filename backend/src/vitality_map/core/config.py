# ==============================================================
#  统一配置：所有环境变量+路径常量+业务阈值集中在这一个地方，
#  取代原来散落在app.py/agent_client.py/llm_client.py各处的
#  os.environ.get(...)调用。其他模块统一 from vitality_map.core.config import settings 用。
# ==============================================================

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 目录结构：backend/src/vitality_map/core/config.py
#   parents[0] = core/          parents[1] = vitality_map/ (PACKAGE_DIR)
#   parents[2] = src/           parents[3] = backend/ (BACKEND_DIR)
#   parents[4] = 仓库根目录 (REPO_ROOT)
#
# 这套推算依赖__file__指向源码原始位置，只在"可编辑安装"(pip install -e，
# Dockerfile里就是这么装的)下成立——普通pip install会把包复制进site-packages，
# __file__不再指向backend/src/，推算就会失效。VITALITY_MAP_DATA_DIR/
# VITALITY_MAP_FRONTEND_DIR这两个环境变量是显式逃生舱：不设置就用下面推算出的
# 默认值，设置了就以环境变量为准，不依赖安装方式。
_THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = _THIS_FILE.parents[1]
BACKEND_DIR = _THIS_FILE.parents[3]
REPO_ROOT = _THIS_FILE.parents[4]

DATA_DIR = Path(os.environ["VITALITY_MAP_DATA_DIR"]) if os.environ.get("VITALITY_MAP_DATA_DIR") else BACKEND_DIR / "data"
FRONTEND_DIR = (
    Path(os.environ["VITALITY_MAP_FRONTEND_DIR"]) if os.environ.get("VITALITY_MAP_FRONTEND_DIR")
    else REPO_ROOT / "frontend"
)

GEOJSON_PATH = DATA_DIR / "grid_data.geojson"
STUDY_AREA_BOUNDARY_PATH = DATA_DIR / "study_area_boundary.geojson"
WEIBO_JSON_PATH = DATA_DIR / "weibo_posts.json"
WEIBO_EMBEDDINGS_PATH = DATA_DIR / "weibo_embeddings.npy"


class Settings(BaseSettings):
    """
    环境变量集中读取（原来分散在各模块用os.environ.get()+各自调用load_dotenv()实现，
    隐式依赖进程启动时CWD是backend/——本地开发和Dockerfile里的WORKDIR凑巧对上，
    但这个假设没写在明面上）。这里显式指定env_file绝对路径，不再依赖CWD。
    """

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek_api_key: str | None = None
    siliconflow_api_key: str | None = None
    qweather_api_key: str | None = None
    tavily_api_key: str | None = None
    amap_api_key: str | None = None

    # /api/chat 限流：每IP每chat_rate_window_seconds秒最多chat_rate_limit次，防止
    # 公网调用真实LLM API被刷爆费用。简单内存计数，进程重启会清零；部署在反向代理
    # 后面的话，需要改成读X-Forwarded-For。
    chat_rate_limit: int = 10
    chat_rate_window_seconds: int = 60

    deepseek_api_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-chat"

    # 与 scripts/weibo_embed.py 用同一个SiliconFlow embeddings模型，保证查询向量和
    # 离线索引向量在同一空间里，见 retrieval/weibo_search.py 顶部注释里的踩坑记录。
    weibo_embed_model_name: str = "BAAI/bge-large-zh-v1.5"
    siliconflow_embeddings_url: str = "https://api.siliconflow.cn/v1/embeddings"
    weibo_similarity_threshold: float = 0.5

    weibo_rerank_model_name: str = "BAAI/bge-reranker-v2-m3"
    siliconflow_rerank_url: str = "https://api.siliconflow.cn/v1/rerank"
    rerank_max_candidates: int = 200

    # 和风天气：新版平台按项目分配专属API Host，免费未认证账号只能拿未来3天预报。
    qweather_host: str = "https://p26vhhq5qq.re.qweatherapi.com"
    qweather_forecast_days: str = "3d"
    wuhan_location_id: str = "101200101"

    tavily_api_url: str = "https://api.tavily.com/search"

    # 高德地图Web服务API：地理编码+路径规划。申请key时要选"Web服务"平台，
    # 不是"Web端(JS API)"。
    amap_geocode_url: str = "https://restapi.amap.com/v3/geocode/geo"
    amap_direction_driving_url: str = "https://restapi.amap.com/v3/direction/driving"
    amap_direction_walking_url: str = "https://restapi.amap.com/v3/direction/walking"
    amap_direction_transit_url: str = "https://restapi.amap.com/v3/direction/transit/integrated"

    # 中国法定节假日+调休数据源（NateScarlet/holiday-cn社区维护，每日自动抓取
    # 国务院公告更新）
    holiday_cn_url_template: str = (
        "https://raw.githubusercontent.com/NateScarlet/holiday-cn/master/{year}.json"
    )

    # agent循环单轮最多跑几步，见agents/single_agent.py顶部注释里的调参记录
    # （8曾经够用，加了finish工具收尾后多占一轮预算，调到10）
    max_agent_steps: int = 10


settings = Settings()

AMAP_DIRECTION_URLS = {
    "driving": settings.amap_direction_driving_url,
    "walking": settings.amap_direction_walking_url,
    "transit": settings.amap_direction_transit_url,
}

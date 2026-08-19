# ==============================================================
#  通用网页搜索（Tavily API），agent兜底工具——微博数据/城市活力模型都
#  查不到的开放性信息用这个
# ==============================================================

import requests

from vitality_map.core.config import settings


def tool_web_search(query: str) -> dict:
    if not settings.tavily_api_key:
        return {"error": "未配置TAVILY_API_KEY环境变量，网页搜索不可用"}
    try:
        resp = requests.post(
            settings.tavily_api_url,
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            json={"query": query, "search_depth": "basic", "max_results": 5},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": f"网页搜索失败：{e}"}
    # 实测Tavily基础检索(search_depth=basic)每条结果本身就有1200~1500字左右，
    # 之前截到200字太狠——像天气40天预报这种表格型页面，前200字大概率还停在
    # 导航菜单，真正的逐日数据在更靠后的位置，截断反而把有用内容漏掉了，导致
    # 模型明明搜到了理论上覆盖到的页面，却因为看到的是被砍掉的片段而判断"查不到"。
    # 1200基本能覆盖Tavily basic模式的完整内容，不会显著增加token成本。
    return {
        "results": [
            {"title": r["title"], "content": (r["content"] or "")[:1200], "url": r["url"]}
            for r in data.get("results", [])[:5]
        ],
    }

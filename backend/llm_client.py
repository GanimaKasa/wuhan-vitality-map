# ==============================================================
#  LLM问答抽象层。
#  已接入DeepSeek API（OpenAI兼容格式）。若未配置DEEPSEEK_API_KEY，
#  或调用失败，自动降级回退到纯模板拼接的mock回答，保证demo始终可用。
#  接口签名固定：answer_question(question, context) -> str，
#  app.py不用感知内部是mock还是真实API。
# ==============================================================

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()  # 读取 backend/.env 里的 DEEPSEEK_API_KEY

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

SYSTEM_PROMPT = (
    "你是“武汉城市活力地图”网站的问答助手。用户会问关于武汉三环内某片区域/某个时段的活力情况，"
    "系统已经用一个多模态深度学习模型算好了预测值和候选格网列表，交给你的context里。"
    "你的任务：用简洁自然的中文口语，把这些结构化数据组织成一段回答，"
    "报告清楚有多少个候选格网、值最高/最低的几个格网的大致预测值和所在行政区。"
    "不要编造context里没有的数字或结论，不要说你是AI，直接给出结果，不需要开场白。"
)


def _build_user_message(question: str, context: dict) -> str:
    return (
        f"用户问题：{question}\n"
        f"结构化上下文（JSON）：{json.dumps(context, ensure_ascii=False)}"
    )


def _chat_completion(system_prompt: str, user_message: str, max_tokens: int = 300, temperature: float = 0.3) -> str:
    """通用DeepSeek调用，未配置key或请求失败时抛异常，由调用方决定降级方式"""
    resp = requests.post(
        DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _deepseek_answer(question: str, context: dict) -> str:
    return _chat_completion(SYSTEM_PROMPT, _build_user_message(question, context), max_tokens=300)


def _mock_answer(question: str, context: dict) -> str:
    """纯模板拼接兜底实现，不调用任何外部API"""
    intent = context.get("intent", {})
    rows = context.get("rows", [])

    if context.get("fallback") or not rows:
        return (
            "抱歉，我没能从问题里识别出具体的时段或区域（比如“武昌区”“夜间”“周末早高峰”这类关键词），"
            "可以换个说法再问一次吗？例如：“洪山区晚上活力怎么样”或“哪里工作日早高峰最热闹”。"
        )

    district = intent.get("district")
    direction = intent.get("direction")
    periods = intent.get("periods") or []

    scope_desc = district if district else "武汉三环内"
    period_desc = "、".join(periods) if periods else "综合时段"
    direction_desc = "活力最高" if direction == "high" else ("活力最低" if direction == "low" else "活力代表性")

    lines = [f"根据模型预测，{scope_desc}在【{period_desc}】{direction_desc}的格网如下："]
    for r in rows[:5]:
        lines.append(f"  · 格网{r['grid_id']}（{r['district']}）：预测值 {r['value']:.2f}")
    if len(rows) > 5:
        lines.append(f"（共匹配到 {len(rows)} 个格网，仅展示前5个，已在地图上高亮）")
    return "\n".join(lines)


def answer_question(question: str, context: dict) -> str:
    """
    question: 用户原始自然语言问题
    context: {
        "intent": {"periods": [...], "district": str|None, "direction": "high"|"low"|None},
        "rows": [ {grid_id, district, period, value}, ... ],
        "fallback": bool,
    }
    """
    if not DEEPSEEK_API_KEY:
        return _mock_answer(question, context)
    try:
        return _deepseek_answer(question, context)
    except Exception as e:
        fallback = _mock_answer(question, context)
        return f"{fallback}\n（提示：真实大语言模型调用失败，已回退到模板回答。错误：{e}）"


KEYWORD_EXPAND_SYSTEM_PROMPT = (
    "你是一个中文搜索关键词扩展助手，服务于一个武汉城市微博文本检索系统。"
    "用户输入一个关键词或短语，你需要联想3-8个语义相关、适合用来做子串匹配的中文短词"
    "（比如输入“武汉旅游”，可以联想“旅游/景点/公园/博物馆/打卡/国家级景点”）。"
    "只返回一个JSON数组，形如[\"词1\",\"词2\"]，不要有任何其他文字、解释或markdown代码块标记。"
)


def expand_keywords(keyword: str) -> list[str]:
    """把用户输入的关键词语义扩展成一组相关词，用于更宽泛地匹配微博文本。
    DeepSeek不可用或返回格式不对时，降级为只用原始关键词。"""
    if not DEEPSEEK_API_KEY:
        return [keyword]
    try:
        raw = _chat_completion(KEYWORD_EXPAND_SYSTEM_PROMPT, keyword, max_tokens=120, temperature=0.5)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        words = json.loads(raw)
        words = [w for w in words if isinstance(w, str) and w.strip()]
        if keyword not in words:
            words.append(keyword)
        return words[:8] if words else [keyword]
    except Exception:
        return [keyword]


GRID_ACTIVITY_SYSTEM_PROMPT = (
    "你是“武汉城市活力地图”网站的助手，负责根据某个格网区域内的微博文本样本，"
    "总结这个地方的人们大致在从事什么类型的活动。"
    "context里会给你该地区的地点类型分布统计和几条抽样微博文本（已做隐私脱敏，不含用户信息）。"
    "请用1-3句简洁自然的中文描述这个地区的活动特征，只依据给定数据，不要编造，不要说你是AI，不需要开场白。"
)


def _mock_activity_summary(place_type_counts: list[tuple]) -> str:
    if not place_type_counts:
        return "该地区暂无可用的微博数据样本。"
    top = place_type_counts[:3]
    desc = "、".join(f"{name}({count}条)" for name, count in top)
    return f"该地区微博样本主要与{desc}相关。"


def summarize_grid_activity(district: str, place_type_counts: list[tuple], sample_texts: list[str]) -> str:
    """
    district: 行政区名
    place_type_counts: [(地点类型, 条数), ...]，按条数降序
    sample_texts: 抽样的微博文本（已脱敏）
    """
    if not DEEPSEEK_API_KEY:
        return _mock_activity_summary(place_type_counts)
    context = {
        "district": district,
        "place_type_distribution": place_type_counts[:10],
        "sample_texts": sample_texts[:15],
    }
    try:
        return _chat_completion(
            GRID_ACTIVITY_SYSTEM_PROMPT,
            f"结构化上下文（JSON）：{json.dumps(context, ensure_ascii=False)}",
            max_tokens=200,
        )
    except Exception:
        return _mock_activity_summary(place_type_counts)

# ==============================================================
#  自然语言问题 -> 结构化查询意图解析
#  纯规则/关键词匹配，不依赖任何外部LLM，供 llm_client mock 使用。
# ==============================================================

LABEL_COLS = [
    "工作日_深夜", "工作日_早高峰", "工作日_日间", "工作日_晚高峰", "工作日_夜间",
    "休息日_深夜", "休息日_早间", "休息日_日间", "休息日_傍晚", "休息日_夜间",
]

# 时段关键词 -> 命中的label_col后缀
PERIOD_KEYWORDS = {
    "深夜": "深夜", "凌晨": "深夜",
    "早高峰": "早高峰", "早间": "早间", "早上": "早间", "上午": "早间",
    "日间": "日间", "白天": "日间", "中午": "日间",
    "晚高峰": "晚高峰", "傍晚": "傍晚", "黄昏": "傍晚",
    "夜间": "夜间", "晚上": "夜间", "夜晚": "夜间",
}
WORKDAY_KEYWORDS = ["工作日", "平时", "上班", "周一", "周二", "周三", "周四", "周五"]
WEEKEND_KEYWORDS = ["休息日", "周末", "双休", "周六", "周日", "假日"]

HIGH_KEYWORDS = ["高", "活力", "热闹", "繁华", "旺", "人多", "最强", "最好", "火"]
LOW_KEYWORDS = ["低", "冷清", "安静", "人少", "最差", "最弱", "萧条", "空"]


def parse_period(question: str):
    """返回匹配到的 label_col 列表（可能因未指定工作日/休息日而同时匹配两个）"""
    suffix_hit = None
    for kw, suffix in PERIOD_KEYWORDS.items():
        if kw in question:
            suffix_hit = suffix
            break
    if suffix_hit is None:
        return []

    is_workday = any(kw in question for kw in WORKDAY_KEYWORDS)
    is_weekend = any(kw in question for kw in WEEKEND_KEYWORDS)

    matched = [c for c in LABEL_COLS if c.endswith(suffix_hit)]
    if is_workday and not is_weekend:
        matched = [c for c in matched if c.startswith("工作日")]
    elif is_weekend and not is_workday:
        matched = [c for c in matched if c.startswith("休息日")]
    return matched


def parse_district(question: str, known_districts):
    for d in known_districts:
        if d and d in question:
            return d
    return None


def parse_direction(question: str):
    """返回 'high' / 'low' / None"""
    high_hit = any(kw in question for kw in HIGH_KEYWORDS)
    low_hit = any(kw in question for kw in LOW_KEYWORDS)
    if high_hit and not low_hit:
        return "high"
    if low_hit and not high_hit:
        return "low"
    return None


def parse_intent(question: str, known_districts):
    """整合解析结果，返回结构化意图字典，供 llm_client 生成回答"""
    periods = parse_period(question)
    district = parse_district(question, known_districts)
    direction = parse_direction(question)
    return {
        "periods": periods,
        "district": district,
        "direction": direction,
    }

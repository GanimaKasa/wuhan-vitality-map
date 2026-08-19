from vitality_map.retrieval.intent import parse_direction, parse_intent, parse_period

KNOWN_DISTRICTS = ["武昌区", "江汉区", "洪山区"]


def test_parse_period_matches_weekday_night():
    assert parse_period("工作日晚上活力怎么样") == ["工作日_夜间"]


def test_parse_period_no_weekday_marker_matches_both():
    cols = parse_period("晚上活力怎么样")
    assert cols == ["工作日_夜间", "休息日_夜间"]


def test_parse_direction_high():
    assert parse_direction("哪里最热闹") == "high"


def test_parse_direction_low():
    assert parse_direction("哪里比较冷清") == "low"


def test_parse_direction_conflicting_keywords_returns_none():
    # 同时命中high/low关键词时视为没有明确方向，不强行猜一个
    assert parse_direction("从热闹到冷清都想看看") is None


def test_parse_intent_district_and_period():
    intent = parse_intent("武昌区晚上活力怎么样", KNOWN_DISTRICTS)
    assert intent["district"] == "武昌区"
    assert "工作日_夜间" in intent["periods"]
    assert "休息日_夜间" in intent["periods"]

# 这几条测试把这个会话里手工反复验证过的行为写成回归用例，不用再靠每次
# 人肉发请求确认："推荐8个打卡点+规划路线"这类问题不该被无关的活力查询
# 结果污染地图高亮、"你刚才说的xxx"这类追问必须走有记忆的agent路径。

from vitality_map.api.chat import _should_use_agent, _suppress_ungrounded_highlight
from vitality_map.retrieval.intent import parse_intent

KNOWN_DISTRICTS = ["武昌区", "江汉区"]


def test_route_planning_question_uses_agent_path():
    intent = parse_intent("推荐8个打卡点并且帮我规划路线", KNOWN_DISTRICTS)
    assert _should_use_agent("推荐8个打卡点并且帮我规划路线", intent) is True


def test_memory_reference_forces_agent_path_even_with_signal():
    # "活力"/"高"命中direction关键词，has_signal=True，但问题在追问历史，
    # 必须走agent路径而不是被误判成走无记忆的快速路径。
    question = "你刚才说的活力最高的那个格网，是第几号来着？"
    intent = parse_intent(question, KNOWN_DISTRICTS)
    assert intent["direction"] == "high"
    assert _should_use_agent(question, intent, has_history=True) is True


def test_memory_reference_without_history_does_not_force_agent_path():
    question = "武昌区活力最高的格网是哪个"
    intent = parse_intent(question, KNOWN_DISTRICTS)
    assert _should_use_agent(question, intent, has_history=False) is False


def test_suppress_highlight_when_question_has_no_vitality_keyword():
    event = {"type": "final", "answer": "...", "highlight_grid_ids": [1, 2, 3]}
    result = _suppress_ungrounded_highlight(event, "推荐8个打卡点并且帮我规划路线")
    assert result["highlight_grid_ids"] == []


def test_keep_highlight_when_question_has_vitality_keyword():
    event = {"type": "final", "answer": "...", "highlight_grid_ids": [1, 2, 3]}
    result = _suppress_ungrounded_highlight(event, "武汉晚上哪里比较热闹")
    assert result["highlight_grid_ids"] == [1, 2, 3]


def test_suppress_highlight_ignores_non_final_events():
    event = {"type": "tool_call", "tool": "get_vitality"}
    result = _suppress_ungrounded_highlight(event, "推荐路线")
    assert result == event

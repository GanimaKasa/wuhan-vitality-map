# 用monkeypatch换掉真实的DeepSeek调用，只测agent循环自己的逻辑（白名单校验、
# ask_user暂停快照、resume续接），不依赖网络/API key，能在CI里跑。

import json

from vitality_map.agents import single_agent


def _fake_tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        ]
    }


def test_finish_grid_ids_filtered_by_whitelist(monkeypatch):
    """模型在finish里编造了一个从没查过的格网编号，应该被过滤掉，只剩真实查过的那个。"""
    responses = [
        _fake_tool_call("get_vitality", {}),
        _fake_tool_call("finish", {"answer": "答案", "highlight_grid_ids": [123, 999]}),
    ]

    def fake_call_deepseek(messages):
        return responses.pop(0)

    monkeypatch.setattr(single_agent, "_call_deepseek", fake_call_deepseek)

    def extract_map_features(tool_name, result):
        if tool_name == "get_vitality":
            return {"seen_grid_ids": [123]}  # 只有123是真的查到过的
        return {}

    events = list(single_agent.run_agent_stream(
        tool_impls={"get_vitality": lambda: {"results": [{"grid_id": 123}]}},
        extract_map_features=extract_map_features,
        question="随便问点什么",
    ))

    final = events[-1]
    assert final["type"] == "final"
    assert final["highlight_grid_ids"] == [123]  # 999被白名单过滤掉了


def test_ask_user_pauses_and_pending_turn_can_resume(monkeypatch):
    """第一次调用ask_user应该暂停并吐出pending_turn快照；恢复时把reply接到
    对应的tool_call_id上继续跑，不需要重新构造整个对话。"""
    call_log = []

    def fake_call_deepseek(messages):
        call_log.append(messages)
        if len(call_log) == 1:
            return _fake_tool_call("ask_user", {"question": "你想去几个地方？"}, call_id="ask_1")
        # 第二次调用（恢复后）：确认reply已经作为tool角色消息接上了
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["tool_call_id"] == "ask_1"
        assert json.loads(messages[-1]["content"])["answer"] == "3个"
        return _fake_tool_call("finish", {"answer": "好的，安排3个地方", "highlight_grid_ids": []})

    monkeypatch.setattr(single_agent, "_call_deepseek", fake_call_deepseek)

    events = list(single_agent.run_agent_stream(
        tool_impls={}, extract_map_features=lambda name, result: {}, question="帮我规划路线",
    ))
    ask_event = events[-1]
    assert ask_event["type"] == "ask_user"
    assert ask_event["pending_turn"]["tool_call_id"] == "ask_1"

    resumed_events = list(single_agent.run_agent_stream(
        tool_impls={}, extract_map_features=lambda name, result: {},
        pending_turn=ask_event["pending_turn"], reply="3个",
    ))
    final = resumed_events[-1]
    assert final["type"] == "final"
    assert final["answer"] == "好的，安排3个地方"


def test_max_steps_exceeded_gives_fallback_answer(monkeypatch):
    """模型一直不调用finish，应该在settings.max_agent_steps步之后给出兜底回答，
    而不是无限循环。"""
    def fake_call_deepseek(messages):
        return _fake_tool_call("get_weather", {"date": "2026-01-01"})

    monkeypatch.setattr(single_agent, "_call_deepseek", fake_call_deepseek)

    events = list(single_agent.run_agent_stream(
        tool_impls={"get_weather": lambda date: {"text_day": "晴"}},
        extract_map_features=lambda name, result: {},
        question="今天天气怎么样",
    ))
    final = events[-1]
    assert final["type"] == "final"
    assert "复杂" in final["answer"]

import math

from vitality_map.tools.routing import _haversine_m, tool_plan_route_order


def test_haversine_zero_distance():
    assert _haversine_m(114.3, 30.5, 114.3, 30.5) == 0


def test_haversine_known_distance_roughly_matches():
    # 武汉黄鹤楼到户部巷，实测约750米，直线距离粗算允许较宽误差范围
    d = _haversine_m(114.302467, 30.544649, 114.298458, 30.547949)
    assert 500 < d < 1200


def test_plan_route_order_requires_at_least_two_points():
    result = tool_plan_route_order([{"name": "A", "lng": 114.3, "lat": 30.5}])
    assert "error" in result


def test_plan_route_order_finds_shorter_order_than_naive():
    # 三点一线：A---C-------B，按输入顺序A/B/C走是绕路，最优顺序应该是A/C/B
    points = [
        {"name": "A", "lng": 114.0, "lat": 30.0},
        {"name": "B", "lng": 114.3, "lat": 30.0},
        {"name": "C", "lng": 114.1, "lat": 30.0},
    ]
    result = tool_plan_route_order(points)
    assert result["order"] == ["A", "C", "B"]


def test_plan_route_order_falls_back_to_greedy_above_eight_points():
    points = [{"name": str(i), "lng": 114.0 + i * 0.01, "lat": 30.0} for i in range(9)]
    result = tool_plan_route_order(points)
    assert len(result["order"]) == 9
    assert math.isfinite(result["total_straight_line_distance_m"])

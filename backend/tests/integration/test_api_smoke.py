# 只测不依赖外部API key的只读端点（geojson/districts/search/all_pois），
# 保证路由能正确挂载、启动时的数据加载不报错。/api/chat、/api/weather这类
# 依赖DeepSeek/和风天气真实key的端点不在这里测——那些属于集成/端到端测试，
# 需要真实凭据，这次重构只保证"结构搬对了"，不重新验证外部API集成本身
# （那部分已经在这次会话里用真实请求反复验证过）。

import pytest
from fastapi.testclient import TestClient

from vitality_map.main import app

client = TestClient(app)


def test_geojson_endpoint_returns_all_grids():
    resp = client.get("/api/geojson")
    assert resp.status_code == 200
    assert len(resp.json()["features"]) == 10011


def test_districts_endpoint():
    resp = client.get("/api/districts")
    assert resp.status_code == 200
    assert len(resp.json()["districts"]) > 0


def test_search_endpoint_ascending_has_no_negative_values():
    resp = client.get("/api/search", params={"order": "asc", "topn": 20})
    assert resp.status_code == 200
    values = [r["value"] for r in resp.json()["results"]]
    assert all(v >= 0 for v in values)


def test_search_endpoint_district_filter():
    resp = client.get("/api/search", params={"district": "武昌区", "topn": 5})
    assert resp.status_code == 200
    for r in resp.json()["results"]:
        assert "武昌区" in r["district"]


def test_all_pois_endpoint():
    resp = client.get("/api/weibo/all_pois")
    assert resp.status_code == 200
    assert resp.json()["count"] > 0


def test_calendar_day_type_rejects_bad_date_format():
    resp = client.get("/api/calendar/day_type", params={"date_str": "not-a-date"})
    assert resp.status_code == 400


def test_frontend_static_files_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]

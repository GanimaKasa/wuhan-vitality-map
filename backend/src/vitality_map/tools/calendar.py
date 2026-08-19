# ==============================================================
#  工作日/休息日判断（含中国法定节假日+调休）
# ==============================================================

from datetime import datetime

import requests
from fastapi import HTTPException

from vitality_map.core.config import settings

# 中国法定节假日+调休数据：用NateScarlet/holiday-cn这个社区维护的开源数据集（每日
# 自动抓取国务院公告更新），比自己按"周六周日=休息"简单判断准确——调休补班日
# （比如国庆调休的周末上班）在这份数据里会被显式标成isOffDay=false。
# 按年缓存在内存里，一年最多请求一次GitHub，避免每次查日期都重新拉取。
_holiday_cache: dict[int, dict] = {}  # year -> {date_str: {"name":..., "is_off_day": bool}}


def _get_holiday_map(year: int) -> dict:
    if year in _holiday_cache:
        return _holiday_cache[year]
    try:
        resp = requests.get(settings.holiday_cn_url_template.format(year=year), timeout=10)
        resp.raise_for_status()
        days = resp.json()["days"]
        holiday_map = {d["date"]: {"name": d["name"], "is_off_day": d["isOffDay"]} for d in days}
    except Exception as e:
        # 拉取失败不缓存空结果——只是这次查询降级为仅按周末判断，下次请求还会重试，
        # 避免一次网络抖动就让这一整年永久锁死在"没有节假日数据"的错误状态。
        print(f"节假日数据拉取失败（{year}年），本次查询降级为仅按周末判断：{e}", flush=True)
        return {}
    _holiday_cache[year] = holiday_map
    return holiday_map


def get_day_type(date_str: str) -> dict:
    """
    判断某天是"工作日"还是"休息日"，含法定节假日/调休（数据源见_get_holiday_map）。
    date_str格式YYYY-MM-DD。节假日数据里没有的日期，退回"周一到周五=工作日，
    周六周日=休息日"的默认规则。
    """
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式应为YYYY-MM-DD")

    holiday_map = _get_holiday_map(d.year)
    entry = holiday_map.get(date_str)
    if entry is not None:
        is_workday = not entry["is_off_day"]
        note = entry["name"] + ("调休上班" if is_workday else "")
    else:
        is_workday = d.weekday() < 5  # 0=周一…4=周五
        note = None

    return {
        "date": date_str,
        "is_workday": is_workday,
        "label": "工作日" if is_workday else "休息日",
        "note": note,
    }


def tool_is_workday(date: str) -> dict:
    """agent工具包装：把HTTPException转成模型能看懂的错误dict，不让异常往上抛炸掉agent循环"""
    try:
        return get_day_type(date_str=date)
    except HTTPException as e:
        return {"error": e.detail}

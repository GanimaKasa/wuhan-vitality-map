# ==============================================================
#  城市活力格网查询（离线推理结果，见 scripts/data_export.py）
# ==============================================================

from vitality_map.core.data import DF, PRED_COLS


def search(district: str | None = None, period: str | None = None,
           topn: int = 20, order: str = "desc") -> dict:
    """
    district: 行政区名（可选，子串匹配）
    period: label_col原始列名（如"工作日_夜间"），对应 pred_<period> 列；缺省用全部10列均值
    topn: 返回条数
    order: asc | desc
    """
    sub = DF
    if district:
        sub = sub[sub["district"].str.contains(district, na=False)]

    ascending = order == "asc"
    if ascending:
        # 最低活力排名不统计水域为主的格网（水域本身没有活力语义，排进"最低"没有意义）
        sub = sub[~sub["is_water"]]

    if period and f"pred_{period}" in sub.columns:
        value_col = f"pred_{period}"
        sub = sub.assign(_value=sub[value_col])
    else:
        sub = sub.assign(_value=sub[PRED_COLS].mean(axis=1))

    sub = sub.sort_values("_value", ascending=ascending).head(topn)

    cols = ["grid_id", "district", "lng", "lat", "_value", "missing_weibo", "missing_streetview"]
    result = sub[cols].rename(columns={"_value": "value"}).to_dict(orient="records")
    return {"count": len(result), "results": result}


def resolve_periods_and_rows(intent: dict) -> list[dict]:
    """快速路径专用：按parse_intent的结果直接查DF，不经过search()的分页/字段裁剪。"""
    periods = intent.get("periods") or []
    district = intent.get("district")
    direction = intent.get("direction")

    sub = DF
    if district:
        sub = sub[sub["district"].str.contains(district, na=False)]

    ascending = direction == "low"
    if ascending:
        sub = sub[~sub["is_water"]]

    if periods:
        value_col = f"pred_{periods[0]}"
    else:
        value_col = None
    sub = sub.assign(val=sub[value_col] if value_col else sub[PRED_COLS].mean(axis=1))

    sub = sub.sort_values("val", ascending=ascending).head(10)

    period_label = periods[0] if periods else "综合时段"
    rows = [
        {"grid_id": int(r.grid_id), "district": r.district, "period": period_label, "value": float(r.val)}
        for r in sub.itertuples()
    ]
    return rows


def tool_get_vitality(period: str | None = None, district: str | None = None,
                       order: str = "desc", topn: int = 10) -> dict:
    result = search(district=district, period=period, topn=topn, order=order)
    return {
        "count": result["count"],
        "results": [
            {"grid_id": r["grid_id"], "district": r["district"], "value": round(r["value"], 2)}
            for r in result["results"]
        ],
    }

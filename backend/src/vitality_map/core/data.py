# ==============================================================
#  启动时一次性加载的静态数据：格网活力预测GeoJSON + 微博脱敏数据 + embedding。
#  原来是app.py顶层的模块级变量，现在单独拆出来，被api/、tools/等各层
#  按需import——避免"哪个模块该拥有这份数据"含糊不清。
# ==============================================================

import json

import numpy as np
import pandas as pd

from vitality_map.core.config import GEOJSON_PATH, STUDY_AREA_BOUNDARY_PATH, WEIBO_EMBEDDINGS_PATH, WEIBO_JSON_PATH

with open(GEOJSON_PATH, encoding="utf-8") as f:
    GEOJSON = json.load(f)

with open(STUDY_AREA_BOUNDARY_PATH, encoding="utf-8") as f:
    STUDY_AREA_BOUNDARY = json.load(f)

_ROWS = []
for feat in GEOJSON["features"]:
    row = dict(feat["properties"])
    row["lng"], row["lat"] = feat["geometry"]["coordinates"]
    _ROWS.append(row)
DF = pd.DataFrame(_ROWS)
KNOWN_DISTRICTS = sorted(DF["district"].dropna().unique().tolist())
GRID_ID_TO_DISTRICT = dict(zip(DF["grid_id"], DF["district"]))
PRED_COLS = [c for c in DF.columns if c.startswith("pred_")]

with open(WEIBO_JSON_PATH, encoding="utf-8") as f:
    WEIBO_DF = pd.DataFrame(json.load(f))
# place_type/post_time缺失时pandas读入会变成float NaN，FastAPI的JSONResponse默认
# allow_nan=False，序列化NaN会直接抛ValueError导致500。这里统一转回None(->JSON null)。
WEIBO_DF["place_type"] = WEIBO_DF["place_type"].astype(object).where(WEIBO_DF["place_type"].notna(), None)
WEIBO_DF["post_time"] = WEIBO_DF["post_time"].astype(object).where(WEIBO_DF["post_time"].notna(), None)

WEIBO_EMBEDDINGS = np.load(WEIBO_EMBEDDINGS_PATH)
assert len(WEIBO_EMBEDDINGS) == len(WEIBO_DF), (
    "weibo_embeddings.npy行数与weibo_posts.json条数不一致，需要重新跑scripts/weibo_embed.py"
)

# 全量POI地图浏览用的轻量字段列表，启动时算一次存着，避免每次请求都重新from_dict转换
# 9.7万行。不带原文（隐私+体积考虑），点位详情复用/api/weibo/grid/{grid_id}按需拉取。
ALL_POI_LIGHT_COLS = ["lng", "lat", "grid_id", "place_type", "like_count"]
ALL_POI_LIGHT = WEIBO_DF[ALL_POI_LIGHT_COLS].to_dict(orient="records")

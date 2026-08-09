# ==============================================================
#  微博原始数据离线预处理：
#  mblogs.csv -> 最近邻匹配到已有的10011个格网(grid_id) -> 脱敏 -> 导出
#
#  隐私处理：只保留 微博文本/经纬度/grid_id/地点类型/发布时间/点赞量，
#  丢弃用户ID/粉丝数/关注数/性别/注册省份/注册城市/注册时间/来源/
#  转发量/评论量/地点ID/微博ID/批次号 等一切可识别用户身份的字段。
#  点赞量是帖子公开互动指标，不识别用户身份，保留用于按热度筛选展示。
# ==============================================================

import json
import os

import pandas as pd
from scipy.spatial import cKDTree

MBLOGS_CSV = r"D:\毕业论文2\原始数据\微博数据\mblogs.csv"
CENTROIDS_CSV = r"D:\毕业论文2\数据采集层\三环线250米\grid_centroids.csv"
OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "weibo_posts.json")

# 与前端HALF_LAT/HALF_LNG同源的格网间隔，取0.75倍作为"落在格网内"的匹配阈值
GRID_SPACING_DEG = 0.00224578
MATCH_THRESHOLD = GRID_SPACING_DEG * 0.75


def main():
    mb = pd.read_csv(MBLOGS_CSV, encoding="utf-8")
    mb = mb.dropna(subset=["经度", "纬度", "微博文本"])

    centroids = pd.read_csv(CENTROIDS_CSV)
    tree = cKDTree(centroids[["centroid_lng", "centroid_lat"]].values)
    dist, idx = tree.query(mb[["经度", "纬度"]].values, k=1)

    within = dist <= MATCH_THRESHOLD
    mb = mb[within].copy()
    mb["grid_id"] = centroids["grid_id"].values[idx[within]]

    posts = []
    for row in mb.itertuples():
        posts.append({
            "text": getattr(row, "微博文本"),
            "lng": float(getattr(row, "经度")),
            "lat": float(getattr(row, "纬度")),
            "grid_id": int(row.grid_id),
            "place_type": getattr(row, "地点类型") if pd.notna(getattr(row, "地点类型")) else None,
            "post_time": getattr(row, "发布时间") if pd.notna(getattr(row, "发布时间")) else None,
            "like_count": int(getattr(row, "点赞量")),
        })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False)

    print(f"原始有效坐标记录：{len(within)}，匹配到三环格网内：{len(posts)}")
    print(f"已导出（脱敏后）到 {OUT_PATH}")


if __name__ == "__main__":
    main()

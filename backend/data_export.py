# ==============================================================
#  全量格网离线推理导出：对10011个格网跑一次模型前向，
#  导出预测活力值(10维) + 8路sigmoid门控权重 + 经纬度/行政区/缺失标记
#  产出 data/grid_data.geojson，供前端直接加载，避免每次请求都跑模型。
#  推理/门控hook写法复用 模型可视化展示/08_plot_gate_weights.py，
#  还原量级写法复用 模型训练层/evaluate_nofin_clip.py。
# ==============================================================

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, r"D:\毕业论文2\模型训练层")
from config import CONFIG
from dataset import VitalityDataset, collate_fn, load_merged_table, standardize_labels
from model import VitalityModel, SEVEN_MODALITY_ORDER
from road_graph import build_road_graph
from modalities import get_gnn_raw_columns

OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "grid_data.geojson")
MODALITY_NAMES = SEVEN_MODALITY_ORDER + ["gnn"]
MODALITY_ZH = ["poi", "road", "landuse", "nightlight", "sentinel2", "weibo", "streetview", "gnn"]
WATER_RATIO_THRESHOLD = 0.5  # lu_ratio_Water_norm超过此值视为水域为主的格网，活力排名（尤其最低排名）应排除


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, groups = load_merged_table()
    graph = build_road_graph()
    grid_id_to_node_idx = {g: i for i, g in enumerate(graph["grid_ids"])}
    edge_index = graph["edge_index"].to(device)
    gnn_raw_cols = get_gnn_raw_columns(groups)
    grid_full = df.set_index(CONFIG["grid_id_col"]).reindex(graph["grid_ids"])
    gnn_raw_all = torch.tensor(grid_full[gnn_raw_cols].values, dtype=torch.float32).to(device)

    ckpt = torch.load(os.path.join(CONFIG["output_dir"], "vitality_model_gradclip_no_fin_clip.pt"),
                       weights_only=False)
    label_stats = ckpt["label_stats"]
    train_mask = df["split"] == "train"
    y_all, _ = standardize_labels(df, CONFIG["label_cols"], train_mask, stats=label_stats)

    all_ds = VitalityDataset(df.reset_index(drop=True), groups, y_all, grid_id_to_node_idx)
    loader = DataLoader(all_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    model = VitalityModel(CONFIG["modality_dims"], n_outputs=len(CONFIG["label_cols"]),
                           use_fin=False, use_cma=True, use_gnn=True).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_gates = []

    def gate_hook(module, inp, output):
        modality_list = inp[0]
        batch_gates = []
        for h, gate_linear in zip(modality_list, module.gates):
            with torch.no_grad():
                g = torch.sigmoid(gate_linear(h))
            batch_gates.append(g.mean(dim=-1).detach().cpu())
        all_gates.append(torch.stack(batch_gates, dim=1))

    hook = model.gate_fusion.register_forward_hook(gate_hook)

    all_pred = []
    all_miss_w, all_miss_sv = [], []
    with torch.no_grad():
        for modal_inputs, missing_mask, node_idx, y in loader:
            modal_inputs = {k: v.to(device) for k, v in modal_inputs.items()}
            missing_mask_dev = missing_mask.to(device)
            pred, _ = model(modal_inputs, gnn_raw_all, edge_index, node_idx.to(device), missing_mask_dev)
            all_pred.append(pred.cpu().numpy())
            all_miss_w.append(missing_mask[:, 5].numpy().astype(bool))
            all_miss_sv.append(missing_mask[:, 6].numpy().astype(bool))

    hook.remove()

    pred_std = np.concatenate(all_pred, axis=0)
    mu = label_stats["mu"].values
    sigma = label_stats["sigma"].values
    pred_raw = pred_std * sigma + mu
    # 真实标签本身非负（百度慧眼热力值求和/取均值，10维标签实测最小值就是0.0），
    # 但回归模型在标准化空间训练，最后一层没有约束输出下限——低活力格网的预测值
    # 偶尔会越过0变成负数，这是test R²=0.4874这个精度水平下越接近0越容易出现的
    # 正常回归噪声，不是bug。裁到0只影响这份导出给网站展示用的数据，不改动
    # checkpoint本身，也不影响已经写进论文的test R²等评估指标（那些指标是在
    # 裁剪之前的pred_raw上算的，这里的裁剪只发生在导出这一步之后）。
    pred_raw = np.clip(pred_raw, 0, None)

    gates = torch.cat(all_gates, dim=0).numpy()
    miss_w = np.concatenate(all_miss_w)
    miss_sv = np.concatenate(all_miss_sv)

    centroids = pd.read_csv(CONFIG["centroids_csv"])
    grid_ids = df[CONFIG["grid_id_col"]].values
    coord_df = pd.DataFrame({CONFIG["grid_id_col"]: grid_ids}).merge(
        centroids[[CONFIG["grid_id_col"], "centroid_lng", "centroid_lat", "行政区"]],
        on=CONFIG["grid_id_col"], how="left",
    )

    water_ratio = df["lu_ratio_Water_norm"].values

    features = []
    for i in range(len(grid_ids)):
        props = {
            "grid_id": int(grid_ids[i]),
            "district": coord_df["行政区"].iloc[i],
            "missing_weibo": bool(miss_w[i]),
            "missing_streetview": bool(miss_sv[i]),
            "water_ratio": round(float(water_ratio[i]), 4),
            "is_water": bool(water_ratio[i] >= WATER_RATIO_THRESHOLD),
        }
        for j, col in enumerate(CONFIG["label_cols"]):
            props[f"pred_{col}"] = round(float(pred_raw[i, j]), 4)
        for j, name in enumerate(MODALITY_ZH):
            props[f"gate_{name}"] = round(float(gates[i, j]), 4)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [coord_df["centroid_lng"].iloc[i], coord_df["centroid_lat"].iloc[i]],
            },
            "properties": props,
        })

    geojson = {"type": "FeatureCollection", "features": features}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    print(f"已导出 {len(features)} 个格网到 {OUT_PATH}")


if __name__ == "__main__":
    main()

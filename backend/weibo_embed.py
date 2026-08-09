# ==============================================================
#  离线计算全部脱敏微博文本的语义embedding，供 /api/weibo/search 做
#  向量余弦相似度检索（替代原来的DeepSeek关键词扩展+字面子串匹配，
#  召回不再受限于扩展词是否恰好出现在原文里）。
#
#  用BAAI/bge-small-zh-v1.5（中文小型embedding模型，~95MB，首次运行
#  从HuggingFace下载），normalize_embeddings=True使得余弦相似度=点积，
#  检索时用numpy矩阵乘法即可，不用每次都算范数。
#
#  产出 weibo_embeddings.npy 的行序必须和 weibo_posts.json 的列表顺序
#  完全一致，app.py按下标对齐两者。
# ==============================================================

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
POSTS_PATH = os.path.join(os.path.dirname(__file__), "data", "weibo_posts.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "weibo_embeddings.npy")


def main():
    with open(POSTS_PATH, encoding="utf-8") as f:
        posts = json.load(f)
    texts = [p["text"] for p in posts]

    # 强制用CPU：GTX1650只有4GB显存，桌面上其他程序已经占用不少，batch编码9万+条
    # 短文本容易OOM；这是一次性离线任务，CPU慢一点但更稳。
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(OUT_PATH, embeddings)
    print(f"已对 {len(texts)} 条微博文本编码，向量维度 {embeddings.shape[1]}，保存到 {OUT_PATH}")


if __name__ == "__main__":
    main()

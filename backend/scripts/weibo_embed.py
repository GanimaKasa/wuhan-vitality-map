# ==============================================================
#  离线计算全部脱敏微博文本的语义embedding，供 /api/weibo/search 做
#  向量余弦相似度检索。
#
#  用SiliconFlow的embeddings API（https://api.siliconflow.cn/v1/embeddings，
#  模型BAAI/bge-large-zh-v1.5，1024维），而不是本地跑sentence-transformers：
#  一是本地GTX1650显存/内存有限跑大批量embedding容易OOM，二是查询时
#  （retrieval/weibo_search.py）也用同一个API算查询向量，两边必须用同一个
#  模型服务保证向量空间完全一致——本地模型和远程API即便"模型名一样"，
#  池化方式可能不同导致向量不可比（之前用HuggingFace免费推理API踩过这个坑）。
#  SiliconFlow返回的是接口自己做好池化的向量，本地和远程都调这同一个
#  接口，从根上避免不一致。
#
#  产出 weibo_embeddings.npy 的行序必须和 weibo_posts.json 的列表顺序
#  完全一致，检索时按下标对齐两者。
#
#  9.7万条文本分批调API，网络抖动/超时在所难免，支持断点续跑：每
#  CHECKPOINT_EVERY个batch落一次盘，中途中断重新运行本脚本会自动从
#  上次进度继续，不用从头再来。
# ==============================================================

import concurrent.futures
import json
import os
import time

import numpy as np
import requests
from dotenv import load_dotenv

# scripts/weibo_embed.py -> 上一级是backend/，.env和data/都在backend/下——
# 显式指定路径，不依赖"必须在backend/目录下手动跑这个脚本"这个隐式假设。
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BACKEND_DIR, ".env"))

MODEL_NAME = "BAAI/bge-large-zh-v1.5"
API_URL = "https://api.siliconflow.cn/v1/embeddings"
API_KEY = os.environ.get("SILICONFLOW_API_KEY")
BATCH_SIZE = 16  # 文档写最多32条，实测这个模型的非Pro版本32条会稳定失败(400)，16条正常
CHECKPOINT_EVERY = 100  # 每100个batch（3200条）落一次盘
MAX_WORKERS = 15  # 并发请求数，SiliconFlow免费额度1000次/分钟，留足余量

POSTS_PATH = os.path.join(BACKEND_DIR, "data", "weibo_posts.json")
OUT_PATH = os.path.join(BACKEND_DIR, "data", "weibo_embeddings.npy")
CHECKPOINT_PATH = OUT_PATH + ".checkpoint.npy"
PROGRESS_PATH = OUT_PATH + ".progress.txt"

EMBED_DIM = 1024  # BAAI/bge-large-zh-v1.5的输出维度，个别文本编码彻底失败时用零向量占位
TRUNCATE_CHARS = 480  # 粗略对应模型512 token上限，超长微博（如转发的长文/多条合并）截断后重试


def _call_api(texts: list[str], max_retries: int = 5):
    """返回embedding数据；请求失败（含网络异常/限流）会重试，仍失败返回None并打印原因"""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"model": MODEL_NAME, "input": texts},
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            last_err = f"网络异常: {e}"
            time.sleep(min(2 ** attempt, 20))
            continue
        if resp.status_code == 429:
            last_err = "429限流"
            time.sleep(min(2 ** attempt, 20))
            continue
        if resp.status_code == 200:
            return resp.json()["data"]
        last_err = f"status={resp.status_code} body={resp.text[:200]}"
        time.sleep(min(2 ** attempt, 20))
    print(f"  请求最终失败（重试{max_retries}次）：{last_err}", flush=True)
    return None


def embed_batch(texts: list[str]) -> np.ndarray:
    data = _call_api(texts)
    if data is not None:
        return np.array([item["embedding"] for item in data], dtype=np.float32)

    # 整批失败：逐条重试，单条还失败就截断再试一次，仍失败则用零向量占位。
    print(f"  批量请求失败，逐条重试（{len(texts)}条）...", flush=True)
    vecs = []
    for text in texts:
        data = _call_api([text], max_retries=3)
        if data is None:
            data = _call_api([text[:TRUNCATE_CHARS]], max_retries=3)
        if data is None:
            print(f"  警告：以下文本编码彻底失败，用零向量占位：{text[:60]}...", flush=True)
            vecs.append([0.0] * EMBED_DIM)
        else:
            vecs.append(data[0]["embedding"])
    return np.array(vecs, dtype=np.float32)


def normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vecs / norms


def main():
    if not API_KEY:
        raise RuntimeError("未配置SILICONFLOW_API_KEY环境变量")

    with open(POSTS_PATH, encoding="utf-8") as f:
        posts = json.load(f)
    texts = [p["text"] for p in posts]

    start_idx = 0
    all_embeddings = []
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            start_idx = int(f.read().strip())
        all_embeddings.append(np.load(CHECKPOINT_PATH))
        print(f"检测到断点，从第{start_idx}条继续（已完成{start_idx}/{len(texts)}）", flush=True)

    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    remaining_starts = list(range(start_idx, len(texts), BATCH_SIZE))
    batches = [texts[i:i + BATCH_SIZE] for i in remaining_starts]

    batches_done_this_run = 0
    start_time = time.time()
    # ThreadPoolExecutor.map按提交顺序yield结果（即使worker间是并发执行/乱序完成），
    # 这样落盘的checkpoint天然是"前N条已全部完成"的连续前缀，不用额外处理乱序。
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for start, raw_vecs in zip(remaining_starts, executor.map(embed_batch, batches)):
            batch_len = min(BATCH_SIZE, len(texts) - start)
            vecs = normalize(raw_vecs).astype(np.float16)
            all_embeddings.append(vecs)
            batches_done_this_run += 1
            batch_idx = start // BATCH_SIZE + 1
            done_count = start + batch_len

            if batches_done_this_run % CHECKPOINT_EVERY == 0:
                checkpoint_arr = np.concatenate(all_embeddings, axis=0)
                np.save(CHECKPOINT_PATH, checkpoint_arr)
                with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
                    f.write(str(done_count))
                elapsed = time.time() - start_time
                rate = batches_done_this_run * BATCH_SIZE / elapsed
                print(f"进度 {batch_idx}/{n_batches}（{done_count}/{len(texts)}条，已落盘，速度{rate:.0f}条/秒）", flush=True)
            elif batch_idx % 20 == 0:
                print(f"进度 {batch_idx}/{n_batches}（{done_count}/{len(texts)}条）", flush=True)

    embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(OUT_PATH, embeddings)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)
    print(f"已对 {len(texts)} 条微博文本编码，向量维度 {embeddings.shape[1]}，保存到 {OUT_PATH}")


if __name__ == "__main__":
    main()

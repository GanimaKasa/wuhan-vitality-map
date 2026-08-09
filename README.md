# 武汉城市活力地图（原型demo）

面向公众的地图问答网站：可视化武汉三环内10011个250m格网的活力预测结果（来自 `模型训练层` 的最终模型 `vitality_model_gradclip_no_fin_clip.pt`），支持地图查询检索、自然语言问答（已接入DeepSeek API），以及基于真实微博文本的关键词搜索/地区活动解读。

## 目录结构
```
backend/
  data_export.py   # 离线推理导出脚本（模型改动后需重新运行）
  weibo_export.py    # 微博原始数据离线预处理：匹配grid_id+脱敏+导出（微博原始数据更新后需重新运行）
  weibo_embed.py      # 微博文本离线embedding计算，供语义检索用（weibo_posts.json更新后需重新运行）
  app.py                # FastAPI服务（含/api/chat、/api/weibo/grid每IP限流；/api/weibo/search不调用LLM不限流）
  llm_client.py          # LLM问答，已接DeepSeek API，无key时自动降级为模板mock
  retrieval.py            # 自然语言意图解析（活力问答用）
  .env                    # 本地环境变量（DEEPSEEK_API_KEY），已在.gitignore里排除
  data/grid_data.geojson    # data_export.py 的产出
  data/weibo_posts.json      # weibo_export.py 的产出（已脱敏）
  data/weibo_embeddings.npy   # weibo_embed.py 的产出（语义检索向量索引）
frontend/            # 纯静态页面（Leaflet地图 + 侧边栏）
requirements.txt
Dockerfile
.gitignore
```

## 本地运行

```bash
# 1. 首次运行 / 模型更新后，重新导出格网数据（需要能访问 D:\毕业论文2\模型训练层 及其checkpoint）
cd backend
python data_export.py

# 1b. 首次运行 / 微博原始数据更新后，重新导出微博数据（需要能访问 D:\毕业论文2\原始数据\微博数据\mblogs.csv）
python weibo_export.py

# 1c. 首次运行 / weibo_posts.json更新后，重新计算语义检索用的embedding
# （首次会从HuggingFace下载BAAI/bge-small-zh-v1.5模型，约95MB；CPU编码9.7万条约12分钟）
python weibo_embed.py

# 2. 安装依赖并启动服务
pip install -r ../requirements.txt
uvicorn app:app --reload --port 8000
```

打开浏览器访问 `http://127.0.0.1:8000` 即可看到地图。

## 大语言模型：DeepSeek API

`backend/llm_client.py` 的 `answer_question()` 已接入 DeepSeek 的 `deepseek-chat` 模型（OpenAI兼容格式，`https://api.deepseek.com/chat/completions`）：

1. 去 [platform.deepseek.com](https://platform.deepseek.com) 申请一个API key。
2. 在 `backend/.env` 里设置：
   ```
   DEEPSEEK_API_KEY=你的key
   ```
   （`.env` 已在 `.gitignore` 里排除，不会被提交进版本库；如果换电脑/环境，需要重新创建这个文件。）
3. 没配置 `DEEPSEEK_API_KEY`，或调用失败（网络问题/key失效/超时），会自动降级回退到纯模板拼接的mock回答（不会导致网站报错）。
4. 如果要换成其他大语言模型（如Anthropic Claude），只需要改 `llm_client.py` 内部的 `_deepseek_answer()` 实现，函数签名 `answer_question(question: str, context: dict) -> str` 不用变，`app.py` 不需要任何改动。

### 费用与限流
- DeepSeek按token计费，这个场景每次问答上下文很小（几百token），单次成本在几厘钱人民币量级，正常使用成本很低。
- 但公网部署后任何人都能调用，为防止被刷爆账单，`app.py` 给 `/api/chat` 加了**每IP每分钟最多10次**的简单内存限流（超限返回429），常量在 `app.py` 顶部的 `CHAT_RATE_LIMIT`/`CHAT_RATE_WINDOW_SECONDS`，可按需调整。注意这个限流是进程内存级别的，重启服务会清零；如果部署在有反向代理/负载均衡的环境，需要改成从 `X-Forwarded-For` 头读真实客户端IP，否则所有请求会被识别成同一个IP。
- 建议同时在DeepSeek后台设置账户消费上限，双重保险。

## 部署（需要用户自行准备云账号）

`Dockerfile` 已提供，可直接构建镜像部署到任意支持Docker的平台（如 Render / Railway / Fly.io）：

```bash
docker build -t wuhan-vitality-map .
docker run -p 8000:8000 wuhan-vitality-map
```

注意：`backend/data/grid_data.geojson` 需要在构建镜像前先在本地生成好（见上面"本地运行"第1步），Dockerfile 会把它一并打包进镜像，容器内不会再重新跑模型推理。

实际域名购买、云主机/平台账号注册、CI/CD配置等运维操作本项目未代做，需要用户自行完成后按上述步骤部署，或后续再协助配置。

## 微博原始文本：语义检索 + 地区活动解读

`原始数据\微博数据\mblogs.csv`（187855条原始记录）经 `weibo_export.py` 离线处理后接入网站，做两个功能：

- **语义检索**（侧边栏"微博热点搜索"面板）：`GET /api/weibo/search?keyword=xxx&top_n=150`。用本地embedding模型（`BAAI/bge-small-zh-v1.5`，`weibo_embed.py`离线算好的`weibo_embeddings.npy`）把查询文本编码成向量，和全部帖子向量算余弦相似度，相似度≥`WEIBO_SIMILARITY_THRESHOLD`（`app.py`里定义，当前0.5）的算候选池——**不再依赖字面关键词匹配**，"黎黄陂路真好玩"这种没提"旅游"两个字但语义相关的帖子也能被搜到。候选池内按`点赞数`（`like_count`）降序取前`top_n`条展示（默认150，前端"按点赞数取前N条展示"输入框可调）。这个接口纯本地计算，不调用DeepSeek，不受限流影响。
  - 效果实测：对"武汉旅游""美食"这类具体名词召回质量很好；对"夜生活"这类抽象复合词，小模型的语义精度有限，候选池会混入一些字面沾边但语义不太相关的内容——如果后续觉得不够准，可以把`app.py`里`WEIBO_EMBED_MODEL_NAME`换成更大的模型（如`bge-base-zh-v1.5`），代价是离线编码和查询都会变慢。
- **地区活动解读**（格网弹窗里的"查看该地区微博动态"按钮）：`GET /api/weibo/grid/{grid_id}`，统计该格网内微博的地点类型分布，抽样几条文本，用 `llm_client.summarize_grid_activity()` 生成一段"这里的人在干什么"的自然语言解读。这个接口调用DeepSeek，复用`/api/chat`的每IP限流保护。

**隐私处理**：`weibo_export.py` 保留 `微博文本`、经纬度、`grid_id`、`地点类型`、`发布时间`、`点赞量`（公开互动指标，不识别用户身份），丢弃了原始数据里的用户ID/粉丝数/关注数/性别/注册地/转发量/评论量等一切可识别用户身份的字段。网站任何地方都不会展示这些被丢弃的字段。

**数据覆盖范围**：原始130741条有效坐标记录里，74.6%（97529条）落在三环格网范围内并保留；其余因超出三环范围被丢弃。

## 已知限制
- LLM问答目前是关键词规则匹配 + 模板生成，不是真正的语言理解，复杂/多轮问题解析能力有限。
- 地图瓦片使用 OpenStreetMap 免key服务，国内访问速度可能一般；如需更好体验可换用高德/百度地图（需要用户自行申请API key）。
- 静态资源（`app.js`/`style.css`）改动后如果浏览器强缓存导致没生效，可以在 `index.html` 里把 `?v=N` 的版本号加一。

# 武汉城市活力地图

面向公众的地图问答网站：可视化武汉三环内10011个250m格网的活力预测结果（来自 `模型训练层` 的最终模型 `vitality_model_gradclip_no_fin_clip.pt`），支持地图查询检索、地图3D可视化、微博语义检索（embedding召回+cross-encoder精排两阶段RAG）、以及一个function calling agent（多轮记忆、中途反问、路线规划）驱动的智能问答。

## 目录结构

后端是标准的Python src-layout布局，业务代码按职责拆成独立的包，取代了早期"所有东西塞进一个app.py"的写法：

```
backend/
  pyproject.toml         # 依赖+打包配置（取代旧的requirements.txt）
  .env                    # 本地环境变量，已在.gitignore里排除
  data/                    # 离线脚本产出的静态数据，运行时直接加载
    grid_data.geojson         # scripts/data_export.py 的产出
    weibo_posts.json           # scripts/weibo_export.py 的产出（已脱敏）
    weibo_embeddings.npy        # scripts/weibo_embed.py 的产出（语义检索向量索引）
  src/vitality_map/       # 运行时代码（部署镜像只需要这里+data/）
    main.py                  # FastAPI入口：只做"实例化app+挂中间件+挂路由+挂静态前端"
    core/                     # 配置(config.py)、启动时加载的共享数据(data.py)、日志、限流
    schemas/                  # Pydantic请求/响应模型
    retrieval/                 # 意图解析(intent.py) + 微博语义检索两阶段RAG(weibo_search.py)
    tools/                      # 可被路由和agent共用的业务逻辑（天气/日历/活力/微博/地理编码/路线）
    services/                    # LLM问答封装（快速路径用，agent见agents/）
    agents/
      single_agent.py            # 手写ReAct循环（"模式A"）：多轮记忆、ask_user暂停恢复、格网白名单校验
    api/                          # FastAPI路由，按资源拆文件
  scripts/                 # 离线一次性脚本，不是运行时代码，依赖torch/geopandas等重依赖
    data_export.py            # 对10011个格网跑一次模型推理，导出grid_data.geojson
    weibo_export.py             # 微博原始数据脱敏+匹配grid_id
    weibo_embed.py                # 微博文本离线embedding计算（SiliconFlow API）
  tests/
    unit/                    # 不依赖外部API的纯逻辑测试
    integration/               # FastAPI TestClient端到端测试（只测不需要外部key的端点）
frontend/                # 纯静态页面（Leaflet 2D地图 + MapLibre/deck.gl 3D柱状图 + 应用坞式侧栏）
.github/workflows/ci.yml   # push/PR时自动跑测试
Dockerfile
```

## 本地运行

```bash
# 1. 首次运行 / 模型更新后，重新导出格网数据（需要能访问 D:\毕业论文2\模型训练层 及其checkpoint）
cd backend/scripts
python data_export.py

# 1b. 首次运行 / 微博原始数据更新后，重新导出微博数据（需要能访问 D:\毕业论文2\原始数据\微博数据\mblogs.csv）
python weibo_export.py

# 1c. 首次运行 / weibo_posts.json更新后，重新计算语义检索用的embedding（调用SiliconFlow API）
python weibo_embed.py

# 2. 安装后端包（可编辑安装：改代码不用重新pip install）
cd ..
pip install -e .

# 3. 启动服务
uvicorn vitality_map.main:app --reload --port 8000
```

打开浏览器访问 `http://127.0.0.1:8000` 即可看到地图。

### 跑测试

```bash
cd backend
pip install -e ".[dev]"
python -m pytest tests/ -v
```

`tests/`目前只覆盖不依赖外部API key的逻辑（意图解析、路线排序、agent循环本身的白名单校验/暂停恢复机制、只读API端点）——依赖DeepSeek/SiliconFlow/和风天气/高德/Tavily真实凭据的调用链目前还是靠手工验证，没有做成CI能跑的端到端测试。

## 大语言模型：DeepSeek API

`backend/src/vitality_map/services/llm_client.py` 的 `answer_question()`（快速路径用）和 `backend/src/vitality_map/agents/single_agent.py`（agent路径用）都接的是 DeepSeek 的 `deepseek-chat` 模型（OpenAI兼容格式）：

1. 去 [platform.deepseek.com](https://platform.deepseek.com) 申请一个API key。
2. 在 `backend/.env` 里设置：
   ```
   DEEPSEEK_API_KEY=你的key
   ```
3. 快速路径没配置 `DEEPSEEK_API_KEY` 时会自动降级回退到纯模板拼接的mock回答；agent路径没配置key时会直接报错提示用户稍后再试（agent循环本身依赖真实的function calling能力，没有等价的mock实现）。
4. 所有环境变量的读取都集中在 `core/config.py` 的 `Settings` 类（pydantic-settings），不再是各模块各自 `os.environ.get(...)`。

### 费用与限流

- DeepSeek按token计费，快速路径每次问答上下文很小，单次成本在几厘钱人民币量级；agent路径因为多轮工具调用，单次问答可能要调用DeepSeek 5~10次，成本更高但仍在可控范围。
- 公网部署后任何人都能调用，为防止被刷爆账单，`/api/chat` 加了**每IP每分钟最多10次**的简单内存限流（超限返回429），常量在 `Settings.chat_rate_limit`/`chat_rate_window_seconds`。这个限流是进程内存级别的，重启服务会清零；如果部署在有反向代理/负载均衡的环境，需要改成从 `X-Forwarded-For` 头读真实客户端IP。
- 建议同时在DeepSeek后台设置账户消费上限，双重保险。

## 部署（需要用户自行准备云账号）

`Dockerfile` 已提供，可直接构建镜像部署到任意支持Docker的平台（如 Render / Railway / Fly.io）：

```bash
docker build -t wuhan-vitality-map .
docker run -p 8000:8000 wuhan-vitality-map
```

注意：
- `backend/data/grid_data.geojson` 需要在构建镜像前先在本地生成好（见上面"本地运行"第1步），Dockerfile 会把它一并打包进镜像，容器内不会再重新跑模型推理。
- Dockerfile用`pip install -e ./backend`（可编辑安装）而不是普通安装——`core/config.py`靠`Path(__file__)`推算`data/`/`frontend/`的绝对路径，普通安装会把包复制进site-packages导致路径推算失效。如果不想依赖这个推算，也可以显式设置`VITALITY_MAP_DATA_DIR`/`VITALITY_MAP_FRONTEND_DIR`环境变量覆盖。

实际域名购买、云主机/平台账号注册等运维操作本项目未代做，需要用户自行完成后按上述步骤部署。

## 微博原始文本：语义检索 + 地区活动解读

`原始数据\微博数据\mblogs.csv`（187855条原始记录）经 `scripts/weibo_export.py` 离线处理后接入网站，做两个功能：

- **语义检索**（侧边栏"微博数据"应用里的"热点搜索"子标签）：`GET /api/weibo/search?keyword=xxx&top_n=150`，具体实现在 `retrieval/weibo_search.py`。两阶段RAG流程：
  1. **召回**：查询文本和全部9.7万条帖子的embedding都调用**SiliconFlow embeddings API**（`BAAI/bge-large-zh-v1.5`，1024维）算，离线索引向量由`scripts/weibo_embed.py`预先批量算好存入`weibo_embeddings.npy`，查询时用同一个API+模型实时算查询向量，两边保证在同一向量空间。算出相似度≥阈值（默认0.5）的算候选池——不依赖字面关键词匹配。
  2. **精排**：候选池送SiliconFlow rerank API（`BAAI/bge-reranker-v2-m3`，cross-encoder联合attention），按真实相关性重排序，避免"很火但字面沾边"的帖子靠点赞数霸榜。
  - **为什么不在本地/Render上跑embedding模型**：本地PyTorch+transformers光加载就要吃掉800MB+内存，超过Render免费档512MB上限；改用云端API后部署环境不用装torch。
  - **为什么查询和离线索引必须用同一个API+模型**：踩过坑——最早查询时调用HuggingFace免费推理接口、离线索引用本地`sentence-transformers`算，两边"模型名一样"但返回的向量完全不在同一空间（池化方式不同），检索结果是垃圾数据。
  - `scripts/weibo_embed.py`支持断点续跑：每处理3200条自动落一次盘，中途中断重新运行会自动从上次进度继续。
- **地区活动解读**（格网弹窗里的"查看该地区微博动态"按钮）：`GET /api/weibo/grid/{grid_id}`，业务逻辑在`tools/weibo.py`的`get_grid_activity()`，统计该格网内微博的地点类型分布，抽样几条文本，用`services/llm_client.py`的`summarize_grid_activity()`生成一段"这里的人在干什么"的自然语言解读。

**隐私处理**：`scripts/weibo_export.py` 只保留 `微博文本`、经纬度、`grid_id`、`地点类型`、`发布时间`、`点赞量`（公开互动指标，不识别用户身份），丢弃了原始数据里的用户ID/粉丝数/关注数/性别/注册地/转发量/评论量等一切可识别用户身份的字段。

**数据覆盖范围**：原始130741条有效坐标记录里，74.6%（97529条）落在三环格网范围内并保留；其余因超出三环范围被丢弃。

## 智能问答agent

`/api/chat`统一入口：能用关键词规则可靠解析的封闭式问题（"武昌区晚上活力怎么样"）走快速路径（单次DeepSeek调用）；开放性问题（"这个周末去哪玩""推荐几个打卡点并规划路线"）走`agents/single_agent.py`的function calling agent循环，具备：

- **多工具调用**：天气、日历（工作日/节假日）、城市活力查询、微博热点语义检索、通用网页搜索（Tavily）、地理编码+路线规划（高德）
- **中途反问（ask_user）**：模型拿不准用户意图时会主动停下来问一句（选项按钮或开放式输入），借鉴LangGraph的interrupt/resume模式——暂停时把完整对话快照原样吐给前端，用户回答后原样带回来续接，不做事后重建猜测
- **多轮记忆**：前端localStorage压缩存最近N轮问答（默认10轮，可调），后端保持无状态
- **格网高亮的双重把关**：格网要不要在地图上高亮，由模型在收尾时显式挑选（不是调用查询工具就自动高亮），并做白名单校验（只能选真实查询到过的编号）+ 用户问题必须包含活力相关关键词的硬性兜底

详细的设计取舍和踩坑记录见项目开发过程中的记忆文档（不在本仓库内）。

## 已知限制

- 地图瓦片使用 OpenStreetMap 免key服务，国内访问速度可能一般；如需更好体验可换用高德/百度地图瓦片（需要用户自行申请API key）。
- 静态资源（`app.js`/`style.css`）改动后如果浏览器强缓存导致没生效，可以在 `index.html` 里把 `?v=N` 的版本号加一。
- 活力预测模型是回归模型，低活力格网偶尔会预测出负值（已在`scripts/data_export.py`导出时裁剪到非负，属于展示层修正，不改动模型本身）。
- 目前没有结构化的可观测性/成本追踪/离线评估体系，agent的回答质量目前靠手工验证，不是自动化评估。

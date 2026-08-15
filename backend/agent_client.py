# ==============================================================
#  Function calling agent循环：反复调用DeepSeek + 工具，直到给出最终答案。
#  只负责编排循环本身（ReAct风格：想→调用→观察→再想），工具的具体实现
#  （怎么查天气、怎么查活力数据）在app.py里，通过tool_impls参数注入进来，
#  避免这个模块跟app.py产生循环import——app.py依赖DF/WEIBO_DF这些大对象，
#  agent_client不需要认识它们，只需要认识"工具名->可调用对象"这个映射。
#
#  设计参考：Anthropic《Building Effective Agents》里的分类，这是最基础的
#  autonomous agent循环（ReAct），不是写死步骤的workflow——因为案例场景
#  （天气好不好决定推荐室内还是室外）本身就是"要根据中间结果动态调整"的，
#  没法提前把所有分支都枚举写死。
# ==============================================================

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# 硬上限，不依赖模型"自觉"停下来——防止意外的死循环调用，控制成本和延迟。
# 5曾经不够用：路线规划这类任务有硬性顺序依赖（先geocode拿到坐标，才能排访问
# 顺序，排完顺序才能逐段查真实路线），哪怕同一轮里能并行打包多个工具调用，
# 光是"发现候选点→排序→逐段查路线"这条链路最少也要4~5轮，5变成了卡在半路
# 触发"问题太复杂"兜底话术、实际上数据都快查完了的情况（线上实测复现过）。
# 8曾经够用，改成finish工具收尾后又不够了：以前模型最后一轮直接输出文字回答，
# 这一轮同时兼顾"给答案"和"结束对话"两件事；现在必须专门再调一次finish，多占
# 一轮预算——8点打卡+路线这种本来就卡在临界值的复杂问题，实测因此又撞到上限
# （数据全查完了，只差最后一次finish调用）。调到10留出这个新增的固定开销。
MAX_AGENT_STEPS = 10

# Render服务器容器默认跑UTC时区，如果直接用date.today()（读服务器本地时间），
# 北京时间0点~8点这段时间UTC日期还停在"前一天"，会让agent把"今天"算错一整天
# （已线上实测踩过这个坑）。武汉/中国场景必须显式用Asia/Shanghai时区。
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def _today_str() -> str:
    return datetime.now(SHANGHAI_TZ).date().isoformat()

AGENT_SYSTEM_PROMPT_TEMPLATE = """你是"武汉城市活力地图"网站的智能推荐助手。用户会问一些开放性问题，
比如"这个周末去哪玩""晚上想找个热闹的地方""帮我规划一条旅游路线"，你需要综合日历、天气、
城市活力预测、微博热点、必要时的网页搜索和地图工具查到的真实数据，给出有依据、说人话的推荐。

今天的日期是{today}。

可用工具：
- is_workday(date)：判断某天是工作日还是休息日（含法定节假日调休）
- get_weather(date)：查询某天的天气，仅支持未来3天（含今天），超出范围工具会返回错误，
  此时不要编造天气，如实告诉用户超出预报范围
- get_vitality(period, district, order, topn)：查询某时段/行政区的活力预测排名，
  period是"工作日_日间"这种格式的时段名（可选），order是desc(活力从高到低)或asc(从低到高)
- search_weibo_hotspots(keyword, top_n)：语义检索微博热点，看看大家在讨论/去哪儿玩，
  返回的是真实脱敏微博样本，不是精确统计
- web_search(query)：通用网页搜索，微博数据/城市活力数据都覆盖不到的开放性信息（比如具体
  某个新开业地点的介绍、网友推荐的打卡点名称）用这个兜底，搜索结果可能不够准确，回答时
  要提醒用户"仅供参考"

规划旅游路线时用这三个工具配合（用户明确要"路线""行程""怎么走"这类需求才需要）：
- geocode(address)：把一个地名（比如从web_search结果里提取出的打卡点名字）转成精确经纬度，
  地名不够精确查不到时会报错，换个更常见的说法再试
- plan_route_order(points)：给多个候选点（每个要有name/lng/lat）排一个访问顺序，返回的是
  直线距离下的最优顺序，不是精确路网距离——只用来决定"先去哪后去哪"这个大致顺序
- route_between(origin_lng, origin_lat, dest_lng, dest_lat, mode)：查两点间的真实路线
  （mode可选driving驾车/walking步行/transit公交地铁），按plan_route_order排好的顺序，
  对相邻两点逐段调用这个工具拿真实路线，不要跳过这一步直接用直线距离当成真实路程告诉用户

结束对话：
- 你必须调用finish(answer, highlight_grid_ids)工具来结束对话、给出最终回答，不要不调用
  任何工具、直接用一段文字结束——那样地图不会有任何反馈。
- answer是给用户看的最终回答内容，口语化、有理有据，直接给结论和具体推荐地点/时段/路线，
  不要罗列"我调用了什么工具"这种过程。
- highlight_grid_ids是可选的格网编号列表：查数据和决定地图上要强调哪些格网是两件独立的
  事——调用get_vitality/search_weibo_hotspots只是获取信息，不代表这些格网就一定要展示。
  等你想清楚了整个答案，再从这一路查到的候选格网里（可以跨多次调用挑选），主动选出真正
  是这轮推荐依据的那些格网，列进highlight_grid_ids；如果这轮回答不需要强调任何格网（比如
  纯路线规划、或者查过的活力/热点数据只是背景参考，没有直接构成你的推荐结论），传空数组，
  不要为了"有内容"而把无关的格网也塞进去。geocode/route_between查到的地点和路线不用你
  操心，会自动显示在地图上。

要求：
- 先想清楚需要哪些信息再决定调用顺序，不要瞎调用不相关的工具
- 工具报错时（比如日期超出天气预报范围、天气/搜索服务不可用、地名查不到坐标）如实告诉用户，
  不要编造数据
- 不要说你是AI，不需要开场白
"""

AGENT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "is_workday",
            "description": "判断某天是工作日还是休息日，含中国法定节假日调休规则",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD格式日期"},
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询武汉某天的天气预报，仅支持未来3天（含今天），超出范围会返回错误",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD格式日期"},
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vitality",
            "description": "查询武汉三环内某时段/行政区的城市活力预测排名（基于多模态深度学习模型）",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": '时段名，如"工作日_日间""休息日_夜间"，不传则用10个时段综合平均',
                    },
                    "district": {"type": "string", "description": "行政区名（可选，如武昌区）"},
                    "order": {
                        "type": "string",
                        "enum": ["desc", "asc"],
                        "description": "desc=活力从高到低（默认），asc=活力从低到高",
                    },
                    "topn": {"type": "integer", "description": "返回条数，默认10"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_weibo_hotspots",
            "description": "语义检索微博热点，查真实脱敏微博用户在讨论/去哪些地方，反映本地人的真实动态",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "检索关键词，如“武汉美食”“夜生活”"},
                    "top_n": {"type": "integer", "description": "返回条数，默认20"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "通用网页搜索，用于微博数据/城市活力数据覆盖不到的开放性信息"
                "（比如具体新地点介绍、超出天气预报范围的天气问题兜底）。"
                "结果来自公开网页，可能不够准确，不是官方权威数据。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": "把一个地名转成精确经纬度坐标（武汉范围内），地名不够精确时查不到会报错",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "地名，如“黄鹤楼”“楚河汉街”"},
                },
                "required": ["address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_route_order",
            "description": "给多个候选打卡点排一个访问顺序（按直线距离最优排序，不是精确路网距离）",
            "parameters": {
                "type": "object",
                "properties": {
                    "points": {
                        "type": "array",
                        "description": "候选点列表，每个点要有name/lng/lat三个字段",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "lng": {"type": "number"},
                                "lat": {"type": "number"},
                            },
                            "required": ["name", "lng", "lat"],
                        },
                    },
                },
                "required": ["points"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_between",
            "description": "查两点间的真实路线（驾车/步行/公交地铁），公交模式含地铁换乘站和出入口信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_lng": {"type": "number"},
                    "origin_lat": {"type": "number"},
                    "dest_lng": {"type": "number"},
                    "dest_lat": {"type": "number"},
                    "mode": {
                        "type": "string",
                        "enum": ["driving", "walking", "transit"],
                        "description": "driving=驾车，walking=步行，transit=公交地铁组合",
                    },
                },
                "required": ["origin_lng", "origin_lat", "dest_lng", "dest_lat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "结束对话并给出最终回答。收集完所有需要的信息、想清楚要怎么回答用户后，"
                "必须调用这个工具，不要不调用任何工具就直接用文字回复结束对话。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "给用户看的最终回答，口语化、有理有据，直接给结论和具体推荐",
                    },
                    "highlight_grid_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "要在地图上高亮的格网编号。从你之前调用get_vitality/"
                            "search_weibo_hotspots等工具查到的候选格网里，挑出真正构成这轮"
                            "推荐依据的那些（可以跨多次调用挑选，不限于最后一次）；如果不需要"
                            "强调任何格网（比如纯路线规划，或者查过的数据只是背景参考、不是"
                            "具体推荐结论），传空数组，不要为了有内容而塞进无关格网。"
                        ),
                    },
                },
                "required": ["answer"],
            },
        },
    },
]

TOOL_DISPLAY_NAMES = {
    "is_workday": "查询工作日/休息日",
    "get_weather": "查询天气",
    "get_vitality": "查询城市活力数据",
    "search_weibo_hotspots": "检索微博热点",
    "web_search": "搜索网页",
    "geocode": "查询地点坐标",
    "plan_route_order": "规划访问顺序",
    "route_between": "查询路线",
}


def _call_deepseek(messages: list[dict]) -> dict:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未配置DEEPSEEK_API_KEY环境变量")
    resp = requests.post(
        DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "tools": AGENT_TOOLS_SCHEMA,
            "temperature": 0.3,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


def run_agent_stream(question: str, tool_impls: dict, extract_map_features):
    """
    question: 用户原始问题
    tool_impls: {工具名: 可调用对象}，由app.py传入，实际执行逻辑在app.py里
    extract_map_features: (tool_name, result_dict) -> {"markers":[...], "polylines":[...]}，
        从工具结果里挑出能在地图上画的东西，由app.py传入（因为哪个字段对应什么地图元素
        跟数据结构强相关）。不再产出grid_ids——格网高亮不是数据查询工具的自动副作用，
        而是模型调用finish时显式挑选的，见下面的说明。

    生成一系列事件dict：
    - {"type": "tool_call", "tool": name, "label": ...}
    - {"type": "tool_result", "tool": name, "label": ...}
    - {"type": "final", "answer": str, "highlight_grid_ids": [...], "markers": [...], "polylines": [...]}

    highlight_grid_ids只来自模型显式调用finish(answer, highlight_grid_ids)时传的参数，
    不再从get_vitality/search_weibo_hotspots的查询结果里自动收集。踩过的坑：早期版本
    是"只要调用了这两个工具，返回里的格网就无条件全部高亮"，后来改成让模型在每次调用时
    传一个show_on_map开关——但这仍然把"要不要展示"这个决策捆在"查询数据的那一刻"，
    模型经常在还没想清楚最终答案前就要预判，容易判断失误（生产环境复现过：agent反复调
    search_weibo_hotspots"找灵感"，每次格网都被展示，跟最终推荐的具体地点毫无关系）。
    现在彻底解耦：get_vitality/search_weibo_hotspots只是单纯的数据查询工具，调用它们
    不会自动产生任何地图效果；等模型想清楚整个答案、调用finish结束对话时，才从它这一路
    看到的候选格网里主动挑选真正要展示的那些。markers/polylines不需要这套机制——
    geocode/route_between的返回结果本身就是"一个具体地点/一段具体路线"，调用即代表
    要展示，没有"顺手查个背景信息"这种歧义，继续在工具执行时自动收集。
    """
    today = _today_str()
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT_TEMPLATE.format(today=today)},
        {"role": "user", "content": question},
    ]
    markers: list[dict] = []
    polylines: list[list] = []

    def _final(answer: str, highlight_grid_ids=None) -> dict:
        return {
            "type": "final",
            "answer": answer,
            "highlight_grid_ids": highlight_grid_ids or [],
            "markers": markers,
            "polylines": polylines,
        }

    for _ in range(MAX_AGENT_STEPS):
        try:
            msg = _call_deepseek(messages)
        except Exception as e:
            yield _final(f"抱歉，推荐服务暂时不可用，请稍后再试。（{e}）")
            return

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            # 模型没有调用finish、直接用文字结束——这是对系统提示词的偏离（属于兜底
            # 容错，不是预期路径），此时没有显式挑选过的格网，highlight_grid_ids留空
            # 比瞎猜要更安全。
            yield _final(msg.get("content") or "抱歉，我没能给出有效回答，换个问法再试试？")
            return

        messages.append(msg)
        for call in tool_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "finish":
                answer = args.get("answer") or "抱歉，我没能给出有效回答，换个问法再试试？"
                raw_grid_ids = args.get("highlight_grid_ids") or []
                grid_ids = [g for g in raw_grid_ids if isinstance(g, int)]
                yield _final(answer, grid_ids)
                return

            label = TOOL_DISPLAY_NAMES.get(name, name)
            yield {"type": "tool_call", "tool": name, "label": label}

            impl = tool_impls.get(name)
            if impl is None:
                result = {"error": f"未知工具：{name}"}
            else:
                try:
                    result = impl(**args)
                except Exception as e:
                    result = {"error": str(e)}

            try:
                features = extract_map_features(name, result) or {}
                markers.extend(features.get("markers", []))
                polylines.extend(features.get("polylines", []))
            except Exception:
                pass

            yield {"type": "tool_result", "tool": name, "label": label}
            # 下划线开头的字段（比如route_between的_polyline，几百个坐标点）只给地图用，
            # 不喂给模型——模型不需要看坐标串，喂了只会白白吃掉token。
            model_visible_result = {k: v for k, v in result.items() if not k.startswith("_")}
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(model_visible_result, ensure_ascii=False),
            })

    yield _final("这个问题有点复杂，我没能在几步内理清楚，要不换个更具体的问法？")

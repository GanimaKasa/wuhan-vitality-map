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
- get_weather(date)：查询某天的天气，仅支持未来3天（含今天），超出范围工具会返回错误。
  这时不要就此放弃或者直接说"查不到"——很多天气网站本身就提供7天/10天/15天的中期预报
  （不是精确预报但也不是历史均值，中央气象台、中国天气网、AccuWeather等都有），应该用
  web_search接着查，查询词要用"<城市>15天天气预报"/"<城市>10天天气预报"这类措辞（不要
  只搜具体日期，比如"武汉8月22日天气"这样搜大概率只搜到历史气候均值网站，搜不到真正的
  中期预报页面）。查到中期预报的具体数字后按预报讲；如果确实只搜到历史气候均值（网站
  明确写"历史均值"字样，或者目标日期超出了搜到的预报页面覆盖范围），才说明这是历史同期
  参考、不是预报。不要不编造精确预报，但也不要动不动就说"查不到准确信息"——先老老实实
  查一遍，查到什么就正面讲什么，说清楚这是"预报"还是"历史参考"，别自己先设限放弃查询
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

中途反问：
- 拿不准该怎么理解用户的意图、或者觉得某个决策最好让用户自己选时，调用ask_user(question,
  options)主动停下来问一句，不要自己瞎猜——比如用户说"推荐几个打卡点"但没说想要几个/什么
  类型，或者你觉得"要不要把活力数据画在地图上"这种事应该让用户自己决定。有明确的几个选项
  可选时，把它们列进options，用户会看到按钮直接点选；没有固定选项、需要用户自己打字回答
  的开放式问题（比如具体地名、数字），options留空。调用后对话会暂停等用户回答，所以只在
  真的没有用户输入就没法继续时才问，能自己合理判断的不要问，不要为了显得严谨而每次都问。

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
  操心，会自动显示在地图上。只能列你自己真的查询到过的格网编号，不要凭印象编造。

对话记忆：
- 如果这次对话历史里已经包含之前几轮的问答（更早的user/assistant消息），直接当成已知
  上下文使用，不用重新自我介绍、也不用把用户之前已经问过的信息再问一遍。
- 用户明确在追问"你刚才说的""上面提到的"这类指向之前某句话的问题时，答案就在历史里
  之前那条assistant消息的文字里，直接读出来回答，不要重新调用工具去查一个新结果——
  哪怕新查出来的结果看起来更"完整"或者你不确定历史里那个数字是不是最新的，也不要在
  用户只是想确认"你之前说的是什么"时用一个新查询的结果替换掉，那样会前后对不上、
  显得前言不搭后语。只有日期/天气这类明确会过期、且用户问的是"现在/最新"而不是"你刚才
  说的"时，才需要重新查一次。

要求：
- 先想清楚需要哪些信息再决定调用顺序，不要瞎调用不相关的工具
- 工具报错、或者超出能力范围（比如日期超出天气预报范围）时，"不编造数据"指的是不要把
  没查到的精确数字说成是精确数字——不代表遇到这种情况就只能说"查不到"。只要你还查到了
  别的相关信息（哪怕不够精确，比如历史同期气候均值、网页搜索里的相关内容），就应该正面
  把这些信息讲出来、说明这是根据什么查到的（"根据查到的资料/历史数据..."），而不是先说
  一句"没有准确信息"再把真正有用的内容当成事后补充。用户想知道你查到了什么，不是想先
  听一句道歉
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
            "name": "ask_user",
            "description": (
                "遇到拿不准该怎么理解用户意图、或者某个决策最好让用户自己选的情况，主动停下来"
                "问用户一句，而不是自己瞎猜。调用后对话会暂停，等用户回答才继续，所以只在真的"
                "需要用户输入才能继续时才用，能自己合理判断的不要问。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要问用户的问题，口语化、简短"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "有明确的几个选项可选时列在这里，用户会看到按钮直接点选；开放式"
                            "问题（比如问具体地名、数字，没法枚举）不传这个字段，用户会自己"
                            "打字回答。"
                        ),
                    },
                },
                "required": ["question"],
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


def _build_fresh_messages(question: str, history: list[dict] | None) -> list[dict]:
    """
    history: [{"question": str, "answer": str}, ...]，前端传来的、已经压缩过的历史轮次
    （只有问题文本+最终答案文本，不含中间工具调用细节——那部分细节太大，且每轮之间
    互不相关，没必要原样带过来）。拼成普通的user/assistant消息对，接到系统提示词后面，
    让模型"记得"之前聊过什么，但不需要重放当时具体调用了哪些工具。
    """
    today = _today_str()
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT_TEMPLATE.format(today=today)}]
    for turn in (history or []):
        q, a = turn.get("question"), turn.get("answer")
        if q and a:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    return messages


def run_agent_stream(
    tool_impls: dict,
    extract_map_features,
    question: str | None = None,
    history: list[dict] | None = None,
    pending_turn: dict | None = None,
    reply: str | None = None,
):
    """
    两种起步方式，二选一：
    - question(+history)：全新一轮（或者上一轮已经正常结束、这轮是紧接着问的新问题）。
      history是之前完成过的轮次的压缩记录，见_build_fresh_messages。
    - pending_turn(+reply)：上一轮被ask_user打断、这次是恢复。pending_turn是上次
      "ask_user"事件里原样吐给前端、又原样传回来的{"messages": [...], "tool_call_id": str}——
      不做任何重建/猜测，只是把reply包装成对应那次工具调用的回应，接上继续跑同一个循环。
      这是借鉴LangGraph interrupt/resume模式的核心思路：暂停时存精确快照，恢复时原样
      接回去，而不是事后猜测该怎么拼消息。

    tool_impls: {工具名: 可调用对象}，由app.py传入，实际执行逻辑在app.py里
    extract_map_features: (tool_name, result_dict) -> {"markers":[...], "polylines":[...],
        "seen_grid_ids":[...]}，由app.py传入。markers/polylines在工具执行时直接累加进
        最终结果；seen_grid_ids只是记录"模型这一路真的查到过哪些格网编号"，用来在finish
        阶段做白名单校验，本身不产生任何地图效果。

    生成一系列事件dict：
    - {"type": "tool_call", "tool": name, "label": ..., "args": {...}}
    - {"type": "tool_result", "tool": name, "label": ..., "result": {...}}——args/result原样
      带给前端，用户可以点开这一步看到具体查了什么、查到了什么（跟Claude Code展示工具
      调用的方式一致），result已经过滤掉了下划线开头的字段（太长/没意义，比如route_between
      的完整坐标串）
    - {"type": "ask_user", "question": str, "options": [...] | None, "pending_turn": {...}}
    - {"type": "final", "answer": str, "highlight_grid_ids": [...], "markers": [...], "polylines": [...]}

    highlight_grid_ids只来自模型显式调用finish(answer, highlight_grid_ids)时传的参数，
    不再从get_vitality/search_weibo_hotspots的查询结果里自动收集（踩过的坑：早期版本
    "只要调用了这两个工具就无条件高亮"、后来的show_on_map开关都还是把"要不要展示"这个
    决策捆在"查询数据的那一刻"，模型经常判断失误）。现在彻底解耦：这两个工具只是单纯的
    数据查询，不产生任何地图效果；等模型调用finish收尾时，才从它这一路查到的候选格网里
    主动挑选。同时做了白名单校验（对照seen_grid_ids）——模型只能选它真的查询到过的编号，
    没查过的一律丢弃，防止编造/记错格网编号却被原样显示到地图上。
    """
    if pending_turn:
        # 原样接回，不重建：messages是上次暂停那一刻的精确快照，reply包装成对那次
        # ask_user调用的工具回应，直接续上。
        messages = list(pending_turn.get("messages") or [])
        tool_call_id = pending_turn.get("tool_call_id")
        if not messages or not tool_call_id:
            yield {
                "type": "final", "answer": "抱歉，这次对话的状态丢失了，麻烦重新问一遍。",
                "highlight_grid_ids": [], "markers": [], "polylines": [],
            }
            return
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps({"answer": reply or ""}, ensure_ascii=False),
        })
    else:
        messages = _build_fresh_messages(question or "", history)

    markers: list[dict] = []
    polylines: list[list] = []
    seen_grid_ids: set[int] = set()

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
                grid_ids = [g for g in raw_grid_ids if isinstance(g, int) and g in seen_grid_ids]
                yield _final(answer, grid_ids)
                return

            if name == "ask_user":
                question_text = args.get("question") or "能再说得具体一点吗？"
                options = args.get("options")
                if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
                    options = None
                # messages此刻已经包含了这条带ask_user调用的助手消息（上面messages.append(msg)
                # 那一步），直接原样打包当快照——不需要再额外处理，见函数开头pending_turn的说明。
                yield {
                    "type": "ask_user",
                    "question": question_text,
                    "options": options,
                    "pending_turn": {"messages": messages, "tool_call_id": call["id"]},
                }
                return

            label = TOOL_DISPLAY_NAMES.get(name, name)
            # args原样带给前端，用户点开这一步能看到"这次查询用了什么参数"（比如geocode
            # 查了哪个地名），跟Claude Code展示工具调用的方式一致。
            yield {"type": "tool_call", "tool": name, "label": label, "args": args}

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
                seen_grid_ids.update(features.get("seen_grid_ids", []))
            except Exception:
                pass

            # 下划线开头的字段（比如route_between的_polyline，几百个坐标点）只给地图用，
            # 不喂给模型（模型不需要看坐标串，喂了只会白白吃掉token）、也不展示给用户
            # （太长了，点开步骤详情只会看到一坨坐标数字，没意义）。
            model_visible_result = {k: v for k, v in result.items() if not k.startswith("_")}
            yield {"type": "tool_result", "tool": name, "label": label, "result": model_visible_result}
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(model_visible_result, ensure_ascii=False),
            })

    yield _final("这个问题有点复杂，我没能在几步内理清楚，要不换个更具体的问法？")

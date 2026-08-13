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
MAX_AGENT_STEPS = 5

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

要求：
- 先想清楚需要哪些信息再决定调用顺序，不要瞎调用不相关的工具
- 工具报错时（比如日期超出天气预报范围、天气/搜索服务不可用、地名查不到坐标）如实告诉用户，
  不要编造数据
- 最终回答要口语化、有理有据，直接给结论和具体推荐地点/时段/路线，不要罗列"我调用了什么工具"
  这种过程
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


def run_agent_stream(question: str, tool_impls: dict, extract_grid_ids):
    """
    question: 用户原始问题
    tool_impls: {工具名: 可调用对象}，由app.py传入，实际执行逻辑在app.py里
    extract_grid_ids: (tool_name, result_dict) -> list[int]，从工具结果里挑出可以在
        地图上高亮的grid_id，由app.py传入（因为哪个字段是grid_id这件事跟数据结构强相关）

    生成一系列事件dict：
    - {"type": "tool_call", "tool": name, "label": ...}
    - {"type": "tool_result", "tool": name, "label": ...}
    - {"type": "final", "answer": str, "highlight_grid_ids": [...]}
    """
    today = _today_str()
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT_TEMPLATE.format(today=today)},
        {"role": "user", "content": question},
    ]
    highlight_grid_ids: list[int] = []

    for _ in range(MAX_AGENT_STEPS):
        try:
            msg = _call_deepseek(messages)
        except Exception as e:
            yield {
                "type": "final",
                "answer": f"抱歉，推荐服务暂时不可用，请稍后再试。（{e}）",
                "highlight_grid_ids": highlight_grid_ids,
            }
            return

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            yield {
                "type": "final",
                "answer": msg.get("content") or "抱歉，我没能给出有效回答，换个问法再试试？",
                "highlight_grid_ids": highlight_grid_ids,
            }
            return

        messages.append(msg)
        for call in tool_calls:
            name = call["function"]["name"]
            label = TOOL_DISPLAY_NAMES.get(name, name)
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

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
                highlight_grid_ids.extend(extract_grid_ids(name, result))
            except Exception:
                pass

            yield {"type": "tool_result", "tool": name, "label": label}
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })

    yield {
        "type": "final",
        "answer": "这个问题有点复杂，我没能在几步内理清楚，要不换个更具体的问法？",
        "highlight_grid_ids": highlight_grid_ids,
    }

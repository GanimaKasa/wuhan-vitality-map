# ==============================================================
#  模式A(single_agent.py)直接用requests裸调DeepSeek REST接口；模式B改用
#  langchain_openai.ChatOpenAI，靠“DeepSeek是OpenAI兼容接口，换base_url就能接”
#  这一点接入LangGraph生态（真实调用已验证过兼容，见项目记忆）。
#
#  base_url从settings.deepseek_api_url（完整REST路径）反推根地址，不额外硬编码
#  一份"https://api.deepseek.com"字符串——避免两处配置各存一份、以后改了忘记同步。
# ==============================================================

from langchain_openai import ChatOpenAI

from vitality_map.core.config import settings


def get_deepseek_chat_model(temperature: float = 0.3) -> ChatOpenAI:
    if not settings.deepseek_api_key:
        raise RuntimeError("未配置DEEPSEEK_API_KEY环境变量")
    base_url = settings.deepseek_api_url.removesuffix("/chat/completions")
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=base_url,
        temperature=temperature,
    )

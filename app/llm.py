"""DeepSeek 客户端封装。

为什么单独一层：换供应商只改这里；重试、超时、参数默认值集中管理，
业务代码只关心 messages 进、content 出。
"""

import httpx2
from openai import AsyncOpenAI

from app.config import settings

_client: AsyncOpenAI | None = None


def _build_http_client() -> httpx2.AsyncClient:
    """显式构造 HTTP 客户端。

    坑：环境里如果存在 ALL_PROXY=socks://...（scheme 少了 "5"），
    httpx 会直接抛 "Unknown scheme for proxy URL"。所以这里不让它读环境变量，
    默认直连；确有需要才用 settings.llm_proxy 显式指定。
    """
    kwargs: dict = {"timeout": 60.0, "follow_redirects": True}
    if settings.llm_proxy:
        kwargs["proxy"] = settings.llm_proxy
    else:
        kwargs["trust_env"] = False  # 不读 HTTP_PROXY / ALL_PROXY
    return httpx2.AsyncClient(**kwargs)


def get_client() -> AsyncOpenAI:
    """懒加载单例：不要在 import 时就建连接，否则测试/离线场景会挂。"""
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            http_client=_build_http_client(),
            max_retries=2,  # SDK 内置重试，针对 429 / 5xx / 连接错误
        )
    return _client


async def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """非流式对话。返回原始 response，让调用方自己取 .choices / .usage。"""
    return await get_client().chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,
    )

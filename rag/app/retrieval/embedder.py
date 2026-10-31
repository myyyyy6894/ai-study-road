"""BGE-M3 embedding 客户端（通过 Ollama 的 HTTP 接口调用）。

为什么用 Ollama 而不是 sentence-transformers：
- 模型加载一次常驻内存，多次调用不用重复加载
- 同一份模型可以被多个项目共享
- 服务端已做归一化，客户端逻辑更简单
"""

import httpx2
import numpy as np

from app.config import settings

OLLAMA_URL = settings.ollama_url
MODEL = settings.embed_model
DIM = settings.embed_dim

# trust_env=False：不读系统代理。本地服务本来就不该走代理，
# 而且机器上的 ALL_PROXY=socks://... 写法不合法，httpx 解析时会直接抛异常。
_client = httpx2.Client(trust_env=False, timeout=120.0)


def embed(texts: list[str]) -> np.ndarray:
    """批量编码，返回 (N, embed_dim) 的 float32 矩阵。

    一次请求传多条，比逐条调用快一个数量级（少 N-1 次网络往返）。
    """
    resp = _client.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": MODEL, "input": texts},
    )
    resp.raise_for_status()

    vectors = np.array(resp.json()["embeddings"], dtype=np.float32)

    # 归一化后，余弦相似度退化为点积，检索时省掉每次除法。
    # Ollama 返回的已经是归一化向量，这里显式再做一次是为了不依赖服务端行为。
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def embed_query(text: str) -> np.ndarray:
    """单条查询：返回一维 (embed_dim,) 向量，可直接与矩阵做点积。"""
    return embed([text])[0]

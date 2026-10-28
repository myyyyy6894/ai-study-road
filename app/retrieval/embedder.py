import httpx2
import numpy as np

QLLAMA_URL = "http://localhost:11434"

MODEL = "bge-m3"
DIM = 1024

_client = httpx2.Client(trust_env=False, timeout=120.0)


def embed(texts:list[str]) -> np.ndarray:
    resp = _client.post(
        f"{QLLAMA_URL}/api/embed",
        json={"model": MODEL, "input": texts},
    )
    resp.raise_for_status()
    vecs = np.array(resp.json()["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

def embed_query(text:str) -> np.ndarray:
    return embed([text])[0]

    
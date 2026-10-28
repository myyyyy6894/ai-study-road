import httpx2
import numpy as np

OLLAMA_URL = "http://localhost:11434"

# 显式关闭环境代理：机器上的 ALL_PROXY=socks://... 写法不合法，
# httpx 解析到就抛 "Unknown scheme"。本地服务本来也不该走代理。
client = httpx2.Client(trust_env=False, timeout=120.0)

def embed(texts: list[str]) -> np.ndarray:
    # 注意接口名：新版批量接口是 /api/embed（不带 s），
    # 返回 {"embeddings": [...]}；旧的 /api/embeddings 只接受单条 prompt
    resp = client.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": "bge-m3", "input": texts},
    )

    resp.raise_for_status()

    return np.array(resp.json()["embeddings"], dtype=np.float32)


sentences = [
    "今天天气真好，阳光明媚",              # 0
    "外面的天气非常不错",                  # 1  ← 与 0 同义
    "股票市场今天大跌",                    # 2  ← 无关
    "贝叶斯公式用于更新后验概率",          # 3
    "用贝叶斯定理可以在新证据下修正概率",  # 4  ← 与 3 同义
    "我想吃火锅",                          # 5
    "晚上去吃四川火锅吧",                  # 6  ← 与 5 同义
    "向量数据库用于相似度检索",            # 7
]

vecs = embed(sentences)
print("向量矩阵形状：", vecs.shape)
print("前5维：", vecs[0][:5])
print("原始长度：", np.linalg.norm(vecs[0]))

# axis=1 按行求长度，keepdims 保留成 (8,1) 才能逐行相除
norms = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

print("归一化后长度：", np.linalg.norm(norms[0]))

sim = norms @ norms.T
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("相似度矩阵：\n")
print(sim)
print("\n每句最像的另一句：")

for i, s in enumerate(sentences):
    # copy() 是方法调用，漏了括号拿到的是方法对象，不能下标访问
    row = sim[i].copy()
    row[i] = -1          # 排除自己
    j = int(row.argmax())
    print(f"  {s}  →  {sentences[j]}  ({row[j]:.3f})")

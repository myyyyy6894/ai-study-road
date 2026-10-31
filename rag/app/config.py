"""应用配置：统一从环境变量 / .env 读取，代码里不再出现硬编码。

所有可调参数集中在这里，方便：
- 换模型 / 换语料时只改一处
- 评估时通过环境变量覆盖
- 部署时用容器环境变量注入
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- LLM（DeepSeek，OpenAI 兼容协议） ----------
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # ---------- Embedding（Ollama 本地服务） ----------
    ollama_url: str = "http://127.0.0.1:11434"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024

    # ---------- 切片 ----------
    chunk_size: int = 400
    chunk_overlap: int = 100

    # ---------- 检索 ----------
    # 阈值是实测标定的：库外问题最高 0.451、库内最低 0.464。
    # 换 embedding 模型或语料后必须重新测量。
    top_k: int = 8
    min_score: float = 0.45

    # ---------- 路径 ----------
    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/uploads")
    index_dir: Path = Path("data/index")

    # ---------- 网络 ----------
    # 留空 = 忽略系统代理直连。如需代理填 "http://127.0.0.1:7897"，
    # 不要填 socks://（httpx 只认 socks5://）。
    llm_proxy: str | None = None


settings = Settings()

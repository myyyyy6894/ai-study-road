"""应用配置：统一从环境变量 / .env 读取，代码里不再出现硬编码。"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DeepSeek（OpenAI 兼容协议）
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"

    # 生成参数
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # 网络：留空 = 忽略系统代理直连（DeepSeek 国内可直连）。
    # 如需代理，填 "http://127.0.0.1:7897"，不要填 socks://。
    llm_proxy: str | None = None


settings = Settings()

import os
from dotenv import load_dotenv
from openai import OpenAI

for key in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "HTTPS_PROXY","all_proxy", "http_proxy", "https_proxy"):
    os.environ.pop(key, None)

load_dotenv()
client = OpenAI(
    api_key = os.environ["DEEPSEEK_API_KEY"],
    base_url = os.environ["DEEPSEEK_BASE_URL"],
)

resp = client.chat.completions.create(
    model = "deepseek-flash",
    messages = [
        {"role": "user", "content": "用一句话说明什么是RAG"}
    ]
)
print(resp.choices[0].message.content)
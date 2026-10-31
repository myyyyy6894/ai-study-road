import os

from dotenv import load_dotenv
from openai import OpenAI

for key in (
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "http_proxy",
    "https_proxy",
):
    os.environ.pop(key, None)

load_dotenv()
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ["DEEPSEEK_BASE_URL"],
)


def ask(messages):
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
    )
    return resp.choices[0].message.content, resp.usage.prompt_tokens


print("=======实验A：保留assistant回复==========")

messages_a = [
    {"role": "user", "content": "3 的平方加 4 的平方等于多少？只回答数字。"},
]

answer, tokens = ask(messages_a)

print(f"第一轮回答: {answer}  (输入 {tokens} tokens)")

messages_a.append({"role": "assistant", "content": answer})
messages_a.append({"role": "user", "content": "把刚才的结果乘以2,只回答数字"})

answer, tokens = ask(messages_a)
print(f"第二轮回答：{answer}  (输入{tokens} tokens)")


print()
print("=======实验B：清除assistant回复========== ")
messages_b = [
    {"role": "user", "content": "3 的平方加 4 的平方等于多少？只回答数字。"},
]

answer, tokens = ask(messages_b)

print(f"第一轮回答：{answer}   (输入{tokens} tokens)")
messages_b.append({"role": "user", "content": "把刚才的结果乘以2，只回答数字"})

answer, tokens = ask(messages_b)

print(f"第二轮回答: {answer}  (输入 {tokens} tokens)")

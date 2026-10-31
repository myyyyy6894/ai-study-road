import os

from dotenv import load_dotenv
from openai import OpenAI

for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(key, None)

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ["DEEPSEEK_BASE_URL"],
)

DIALOG = """客户：我上周买的耳机，左耳没声音了，能换货吗？
客服：可以的，请提供订单号。
客户：订单号是 20241105,另外我想问下保修多久。
客服：一年保修"""


def ask(prompt):
    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ===== A：模糊指令 =====
prompt_a = f"分析下面这段对话：\n{DIALOG}"

# ===== B：明确任务 + 字段 + 输出格式 =====
prompt_b = f"""从下面的客服对话中抽取信息，输出一个 JSON 对象，包含这些字段：
- product：商品名
- issue：问题描述
- order_id：订单号
- warranty：保修期

只输出 JSON，不要任何解释。

对话：
\"\"\"
{DIALOG}
\"\"\""""

# ===== C：明确任务 + few-shot 示例 =====
prompt_c = f"""从客服对话中抽取信息，输出 JSON。

示例输入：
\"\"\"
客户：我买的充电器坏了
客服：请提供订单号
客户：A123
\"\"\"
示例输出：
{{"product": "充电器", "issue": "损坏", "order_id": "A123", "warranty": null}}

现在处理：
\"\"\"
{DIALOG}
\"\"\""""
print("========== A 模糊指令 ==========")
print(ask(prompt_a))

print("\n========== B 明确任务 + 格式 ==========")
print(ask(prompt_b))

print("\n========== C 明确任务 + few-shot ==========")
print(ask(prompt_c))

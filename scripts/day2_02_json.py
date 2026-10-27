import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel,ValidationError

for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy","HTTPS_PROXY", "https_proxy"):
    os.environ.pop(key, None)

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ["DEEPSEEK_BASE_URL"],
)

def safe_parse(text:str)->dict:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start,end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise

class Extracted(BaseModel):
    product:str
    issue:str
    order_id:str | None = None
    warranty:str | None = None

print("=" * 20, "safe_parse 容错测试", "=" * 20)

test_cases = [
    '{"product": "耳机"}',
    '```json\n{"product": "耳机"}\n```',
    '好的，结果如下：\n{"product": "耳机"}\n希望有帮助',
    'this is not json',
]
for i, case in enumerate(test_cases, 1):
    print(f"\n用例 {i} 输入: {case!r}")
    try:
        print("         结果:", safe_parse(case))
    except Exception as e:
        print("         异常:", type(e).__name__, e)

DIALOG = """客户：我上周买的耳机，左耳没声音了，能换货吗？
客服：可以的，请提供订单号。
客户：订单号是 20241105，另外我想问下保修多久。
客服：一年保修。"""

PROMPT = f"""从下面的客服对话中抽取信息，以 json 对象输出，字段为
product, issue, order_id, warranty。

对话：
\"\"\"
{DIALOG}
\"\"\""""
def extract(use_json_mode: bool) -> None:
    # 根据是否开启 JSON 模式，决定要不要给 API 传 response_format 参数
    # - 开启：让模型强制输出合法 JSON 对象（OpenAI 兼容接口的约定）
    # - 不开启：kwargs 为空字典，什么都不传，走普通文本生成
    # 注意：原代码 "tyoe" 是笔误，应为 "type"；且 if/else 要写在同一行或用括号包住
    kwargs = {"response_format": {"type": "json_object"}} if use_json_mode else {}

    # 单纯为了在控制台打印标题，方便区分两种模式下的输出
    mode = "JSON mode" if use_json_mode else "普通模式"
    print(f"\n{'=' * 20}{mode}{'=' * 20}")

    # 调用 DeepSeek 的 Chat Completion 接口
    resp = client.chat.completions.create(
        model="deepseek-flash",                       # 指定模型
        messages=[{"role": "user", "content": PROMPT}],  # 只发一条 user 消息，PROMPT 是外部定义好的提示词
        temperature=0.2,                              # 温度调低，让输出更稳定、更"照格式走"
        **kwargs,                                     # 把上面的 response_format（可能为空）展开传进去
    )

    # 从返回结构里取出模型实际输出的文本
    # choices[0] 是第一个候选答案，message.content 是文本内容
    raw = resp.choices[0].message.content

    # 用 repr 打印，能看清换行符 \n、引号、是否被代码块包裹等"脏数据"细节
    print("原始输出", repr(raw))

    # 调用前面写的容错解析函数，尽量把 raw 解析成 dict
    data = safe_parse(raw)
    print("解析结果", data)

    # 用 Pydantic 模型 Extracted 做字段级校验（类型、必填、取值范围等）
    try:
        obj = Extracted(**data)               # 把 dict 拆成关键字参数喂给 Pydantic 模型
        print("Pydantic校验通过：", obj.model_dump())   # model_dump() 把模型转回普通 dict 方便查看
    except ValidationError as e:              # 校验失败（缺字段、类型不对等）会抛这个异常
        print("Pydantic 校验失败")
        print(e)


# 分别跑一次"普通模式"和"JSON 模式"，对比两种情况下：
#   1. 原始输出长什么样（有无多余文字、有无代码块包裹）
#   2. safe_parse 能不能解析成功
#   3. Pydantic 校验是否通过
extract(use_json_mode=False)
extract(use_json_mode=True)
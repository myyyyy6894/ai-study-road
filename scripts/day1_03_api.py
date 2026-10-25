import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

for key in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "HTTPS_PROXY","all_proxy", "http_proxy", "https_proxy"):
    os.environ.pop(key, None)

load_dotenv()
client = OpenAI(
    api_key = os.environ["DEEPSEEK_API_KEY"],
    base_url = os.environ["DEEPSEEK_BASE_URL"],
)
app = FastAPI(title = "多轮对话的Demo")

 #会话存储
sessions:dict[str,list[dict]] = {}

SYSTEM = {"role":"system","content":"你是一个简洁的中文助手，回答不超过两句话"}

class ChatIn(BaseModel):
    session_id:str = "default"
    question:str
class ChatOut(BaseModel):
    answer:str
    turns:int
    prompt_tokens:int
@app.post("/api/chat", response_model=ChatOut)

def chat(req:ChatIn) -> ChatOut:
    messages = sessions.setdefault(req.session_id,[SYSTEM])
    messages.append({"role":"user","content":req.question})

    resp = client.chat.completions.create(
        model = "deepseek-flash",
        messages = messages,
    )
    answer = resp.choices[0].message.content
    messages.append({"role":"assistant","content":answer})

    return ChatOut(
        answer = answer,
        turns = len(messages) // 2,
        prompt_tokens = resp.usage.prompt_tokens,
    )

@app.get("/api/session/{session_id}")
def get_session(session_id:str):
    return {"session_id":session_id,"messages":sessions.get(session_id,[])}

@app.delete("/api/sessions/{session_id}")
def clear_session(session_id:str):
    sessions.pop(session_id,None)
    return {"OK",True}

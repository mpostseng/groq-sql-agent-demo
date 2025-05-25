from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.chains import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
import os

# 讀取環境變數
DB_URL = os.getenv("DB_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 初始化 FastAPI
app = FastAPI()

# 使用 Groq API 的 LLM（Mixtral 或 LLaMA3）
llm = ChatOpenAI(
    temperature=0,
    model="mixtral-8x7b-32768",  # 或 "llama3-70b-8192"
    openai_api_key=GROQ_API_KEY,
    openai_api_base="https://api.groq.com/openai/v1"
)

# 初始化資料庫連線
db = SQLDatabase.from_uri(DB_URL)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# 請求格式
class QuestionRequest(BaseModel):
    question: str

# 路由
@app.post("/ask")
async def ask_question(req: QuestionRequest):
    try:
        result = db_chain.run(req.question)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

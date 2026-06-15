# ============================================================
# 2.1.1 模型调用（ChatOpenAI）：单轮对话
# ChatModel 接收角色消息列表，返回一条回复
# ============================================================
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 1. 加载 API 密钥
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")

# 2. 初始化对话模型
chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3,     # 0=严谨, 1=创造
    max_tokens=200       # 限制输出长度
)

# 3. 构造消息列表（三个角色各有作用）
messages = [
    {"role": "system", "content": "你是一个耐心的AI学习助手，回复简洁易懂，适合高校学生理解。"},
    {"role": "user", "content": "请用3句话解释什么是LangChain？"}
]

# 4. 调用模型
result = chat_model.invoke(messages)

# 5. 输出结果
print("ChatModel回复：")
print(result.content)

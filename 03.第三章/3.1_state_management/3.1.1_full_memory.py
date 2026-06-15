import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# 加载环境变量（确保.env文件中配置了API_KEY）
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 初始化LLM模型
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3  # 降低随机性，保证输出稳定
)

# 1. 定义提示词模板（包含历史消息占位符）
full_memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是友好的对话助手，需基于完整的历史对话回答用户问题。"),
    MessagesPlaceholder(variable_name="chat_history"),  # 历史消息占位符
    ("human", "{user_input}")  # 用户当前输入
])

# 2. 构建基础链（提示词 + LLM）
base_chain = full_memory_prompt | llm

# 3. 会话历史存储（内存模式，生产环境可替换为数据库存储）
full_memory_store = {}

# 4. 定义会话历史获取函数（核心：返回完整历史）
def get_full_memory_history(session_id: str) -> BaseChatMessageHistory:
    """根据session_id获取会话历史，不存在则创建新的历史记录"""
    if session_id not in full_memory_store:
        full_memory_store[session_id] = InMemoryChatMessageHistory()
    return full_memory_store[session_id]

# 5. 构建带全量记忆的对话链
full_memory_chain = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_full_memory_history,
    input_messages_key="user_input",  # 输入中用户问题的键名
    history_messages_key="chat_history"  # 传入提示词的历史消息键名
)

# 测试多轮对话（指定session_id=user_001，隔离不同用户）
config = {"configurable": {"session_id": "user_001"}}

# 第一轮对话
response1 = full_memory_chain.invoke({"user_input": "我叫小明，喜欢编程"}, config=config)
print("助手回复1：", response1.content)
# 输出示例：你好小明！编程是一项很有创造力的技能，你平时常用什么编程语言呢？

# 第二轮对话（验证记忆：询问历史信息）
response2 = full_memory_chain.invoke({"user_input": "我刚才说我喜欢什么？"}, config=config)
print("助手回复2：", response2.content)
# 输出示例：你刚才说你喜欢编程呀～

# 查看完整历史记录
print("\n全量记忆的对话历史：")
for msg in get_full_memory_history("user_001").messages:
    print(f"{msg.type}: {msg.content}")
    
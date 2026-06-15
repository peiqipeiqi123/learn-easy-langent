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

# 1. 定义提示词模板（与全量记忆通用，可复用）
window_memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是友好的对话助手，需基于最近的对话历史回答用户问题。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# 2. 构建基础链
window_base_chain = window_memory_prompt | llm

# 3. 会话历史存储
window_memory_store = {}
WINDOW_SIZE = 2  # 保留最近2轮对话（即最近4条消息：用户-助手-用户-助手）

# 4. 定义带窗口限制的会话历史获取函数
def get_window_memory_history(session_id: str) -> BaseChatMessageHistory:
    """获取会话历史，仅保留最近WINDOW_SIZE轮对话"""
    if session_id not in window_memory_store:
        window_memory_store[session_id] = InMemoryChatMessageHistory()
    
    # 获取完整历史，截取最近WINDOW_SIZE轮（每轮2条消息）
    history = window_memory_store[session_id]
    if len(history.messages) > 2 * WINDOW_SIZE:
        # 截取后WINDOW_SIZE轮消息（保留最新的）
        history.messages = history.messages[-2 * WINDOW_SIZE:]
    return history

# 5. 构建带窗口记忆的对话链
window_memory_chain = RunnableWithMessageHistory(
    runnable=window_base_chain,
    get_session_history=get_window_memory_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"
)

# 测试多轮对话（session_id=user_002，与全量记忆会话隔离）
config = {"configurable": {"session_id": "user_002"}}

# 模拟5轮对话，验证窗口记忆的截断效果
inputs = [
    "我叫小红",
    "我喜欢画画",
    "我来自上海",
    "我是一名学生",
    "我刚才说我来自哪里？",  # 第5轮：询问第3轮的信息，验证窗口截断
    "我叫什么名字？"  # 第6轮：询问第1轮的信息，验证窗口记忆
]

for i, user_input in enumerate(inputs, 1):
    response = window_memory_chain.invoke({"user_input": user_input}, config=config)
    print(f"\n第{i}轮 - 助手回复：", response.content)

# 查看窗口记忆的最终历史（仅保留最近2轮）
print("\n窗口记忆的最终对话历史（最近2轮）：")
for msg in get_window_memory_history("user_002").messages:
    print(f"{msg.type}: {msg.content}")
    
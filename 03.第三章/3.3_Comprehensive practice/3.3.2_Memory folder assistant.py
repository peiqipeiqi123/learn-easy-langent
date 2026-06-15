# =========================
# 1. 基础依赖
# =========================
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.messages import AIMessage, ToolMessage

# =========================
# 2. 环境变量 & 模型
# =========================
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

# =========================
# 3. 窗口记忆
# =========================
WINDOW_SIZE = 3
memory_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = InMemoryChatMessageHistory()

    history = memory_store[session_id]
    if len(history.messages) > 2 * WINDOW_SIZE:
        history.messages = history.messages[-2 * WINDOW_SIZE:]
    return history

# =========================
# 4. 定义工具（@tool）
# =========================
@tool
def list_files(path: str = ".") -> str:
    """查看指定目录下的文件列表"""
    try:
        if not os.path.exists(path):
            return f"路径不存在：{path}"

        items = os.listdir(path)
        if not items:
            return "目录为空"

        result = []
        for item in items:
            full = os.path.join(path, item)
            if os.path.isfile(full):
                result.append(f"文件：{item}（{os.path.getsize(full)} 字节）")
            else:
                result.append(f"文件夹：{item}")
        return "\n".join(result)
    except Exception as e:
        return f"查看失败：{e}"

@tool
def create_file(path: str, content: str = "") -> str:
    """创建文件，并可写入初始内容"""
    try:
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"文件已创建：{path}"
    except Exception as e:
        return f"创建失败：{e}"

@tool
def write_file(path: str, content: str, append: bool = True) -> str:
    """向文件写入内容，支持追加或覆盖"""
    try:
        if not os.path.exists(path):
            return f"文件不存在：{path}"

        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)

        return f"写入成功（{'追加' if append else '覆盖'}）"
    except Exception as e:
        return f"写入失败：{e}"

@tool
def delete_file(path: str) -> str:
    """删除文件或空文件夹"""
    try:
        if not os.path.exists(path):
            return f"路径不存在：{path}"

        if os.path.isfile(path):
            os.remove(path)
            return f"文件已删除：{path}"

        if os.path.isdir(path):
            if os.listdir(path):
                return "文件夹非空，无法删除"
            os.rmdir(path)
            return f"文件夹已删除：{path}"

        return "无效路径"
    except Exception as e:
        return f"删除失败：{e}"

tools = [list_files, create_file, write_file, delete_file]

# =========================
# 5. Prompt（告诉模型：你可以用工具）
# =========================
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个文件操作智能助手。"
     "当用户请求涉及文件或目录操作时，你可以自主决定是否调用工具。"
     "如果不需要工具，直接回答用户。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# =========================
# 6. 构建 Tool-Calling Agent
# =========================
agent = prompt | llm.bind_tools(tools)

agent_with_memory = RunnableWithMessageHistory(
    runnable=agent,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

if __name__ == "__main__":
    session_id = "tool_agent_demo"

    print("===== 🧠 Tool Calling 文件 Agent =====")
    print("示例：")
    print(" - 查看当前文件夹")
    print(" - 创建文件 test.txt 内容 Hello")
    print(" - 写入文件 test.txt 内容 World 追加")
    print(" - 删除文件 test.txt")
    print("输入 q 退出\n")

    while True:
        user_input = input("你：")
        if user_input.lower() in ["q", "quit", "退出"]:
            print("助手：再见 👋")
            break

        # ===== 第一次：模型思考 =====
        result = agent_with_memory.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        history = get_session_history(session_id)

        print("\n🧠【模型输出】")
        if result.content:
            print(result.content)

        # ===== 模型决定调用工具 =====
        if isinstance(result, AIMessage) and result.tool_calls:
            print("\n🔧【模型决定调用工具】")
            for call in result.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]

                print(f"➡️ 工具名：{tool_name}")
                print(f"➡️ 参数：{tool_args}")

                tool_func = next(t for t in tools if t.name == tool_name)
                observation = tool_func.invoke(tool_args)

                print("\n📦【工具执行结果】")
                print(observation)

                history.add_message(
                    ToolMessage(
                        tool_call_id=call["id"],
                        content=str(observation)
                    )
                )

            print("\n✅【本轮结束：工具执行完成】\n")
            continue  # 回到 while True 等用户输入

        # ===== 最终回答（没有工具调用） =====
        print("\n✅【最终回答】")
        print(result.content, "\n")
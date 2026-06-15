from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import FileManagementToolkit
from dotenv import load_dotenv
import os

# -------------------
# 1. 初始化环境
# -------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3,
)

# -------------------
# 2. 创建文件管理工具
# -------------------
toolkit = FileManagementToolkit(root_dir=".")
tools = toolkit.get_tools()

# -------------------
# 3. 创建 Agent（最新版）
# -------------------
agent = create_agent(
    model=llm,
    tools=tools,
    debug=True,  # 打开调试，显示模型思考和工具调用过程
)

# -------------------
# 4. 执行任务
# -------------------
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "请创建一个名为 llm诗词.txt 的文件，并在文件中写入一首原创七言绝句，主题围绕科技与人文的融合。"}
    ]
})

print("\n任务执行完成！文件已写入。")
print("Agent最终输出：\n", response["messages"][-1].content)
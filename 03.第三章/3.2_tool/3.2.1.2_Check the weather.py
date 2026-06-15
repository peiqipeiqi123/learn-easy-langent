from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# ======================
# 1. 环境
# ======================
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3,
)

# ======================
# 2. 工具
# ======================
@tool
def weather_query(city: str) -> str:
    """查询指定城市天气"""
    weather_data = {
        "北京": "北京今日天气：晴，-2~8℃",
        "上海": "上海今日天气：多云，5~12℃",
        "兰州": "兰州今日天气：晴，19~31℃",
    }
    return weather_data.get(city, f"暂无 {city} 数据")

tools = [weather_query]

# ======================
# 3. 创建 Agent（开启 debug）
# ======================
agent = create_agent(
    model=llm,
    tools=tools,
    debug=True,  # 👈 打开过程打印
)

# ======================
# 4. 运行
# ======================
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "兰州今天的天气怎么样？"}
    ]
})

print("\n最终回答：")
print(response["messages"][-1].content)
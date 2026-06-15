# ============================================================
# 2.3.1 StrOutputParser：AIMessage → 纯字符串
# 核心：把模型的 AIMessage 对象转成 str，方便后续处理
# 链式调用: llm | parser
# ============================================================
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# 1. 环境初始化
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 2. 初始化模型
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

# 3. 创建解析器
parser = StrOutputParser()

# 4. 链式调用：llm 先调模型 → parser 把结果转成字符串
chain = llm | parser
result = chain.invoke("请简要介绍 LangChain 输出解析层的作用")

print("StrOutputParser 解析后的字符串：")
print(result)
print("\n解析结果类型：", type(result))  # <class 'str'>

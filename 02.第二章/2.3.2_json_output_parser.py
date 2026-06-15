# ============================================================
# 2.3.2 JsonOutputParser：模型输出 → Python 字典
# 核心：自动引导模型输出 JSON，解析为 dict
# 链式调用: prompt | llm | parser
# ============================================================
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 1. 环境与模型初始化
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

# 2. 创建 JSON 解析器
parser = JsonOutputParser()

# 3. 构建提示模板（parser.get_format_instructions() 自动告诉模型输出 JSON）
prompt = PromptTemplate(
    template="请介绍1个LangChain开发工具，输出工具名和核心功能。{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. 链式调用：提示 → 模型 → 解析
chain = prompt | llm | parser
result = chain.invoke({})

print("解析后的JSON（Python字典）：")
print(result)
print("获取单个字段：", result.get('tool_name', None))

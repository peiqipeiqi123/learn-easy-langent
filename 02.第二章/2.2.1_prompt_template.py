# ============================================================
# 2.2.1 提示词模板（PromptTemplate）基础用法
# 核心：把固定文本和动态参数分离，模板→format()→完整提示词
# ============================================================
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")

chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3,
    max_tokens=200
)

# 1. 定义模板：{user_role} 和 {subject} 是动态参数（可替换的占位符）
prompt_template = PromptTemplate(
    input_variables=["user_role", "subject"],
    template="请给{user_role}写一段50字左右的{subject}学习建议，语言简洁实用，分2个小要点。"
)

# 2. 用 format() 填入具体参数 → 生成完整提示词
formatted_prompt = prompt_template.format(
    user_role="高校学生",
    subject="LangChain"
)
print("格式化后的提示词：")
print(formatted_prompt)

# 3. 调用模型
result = chat_model.invoke([{"role": "user", "content": formatted_prompt}])
print("\n生成的学习建议：")
print(result.content)

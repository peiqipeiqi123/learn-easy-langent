# ============================================================
# 2.2.2 少样本提示模板（FewShotPromptTemplate）
# 核心：给模型看几个示例，让它照着示例的格式生成内容
# ============================================================
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
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

# 1. 准备示例（给模型看的"范本"）
examples = [
    {
        "subject": "Python编程",
        "method": "核心目标：掌握基础语法和常用库；学习步骤：1. 学习变量、函数等基础语法 2. 实操小项目（如计算器） 3. 学习Pandas、Matplotlib库；注意事项：多动手实操，遇到错误及时调试。"
    },
    {
        "subject": "机器学习",
        "method": "核心目标：理解基础算法原理和应用场景；学习步骤：1. 复习数学基础（线性代数、概率） 2. 学习经典算法（线性回归、决策树） 3. 用Scikit-learn实操；注意事项：先理解原理，再动手实现，避免死记硬背。"
    }
]

# 2. 定义示例模板：告诉模型如何"阅读"这些示例
example_prompt = PromptTemplate(
    input_variables=["subject", "method"],
    template="学科：{subject}\n学习方法：{method}\n"
)

# 3. 构建少样本提示模板
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,               # 给模型看的参考案例
    example_prompt=example_prompt,    # 示例的格式模板
    suffix="学科：{new_subject}\n学习方法：",  # 最后追加的用户需求
    input_variables=["new_subject"]
)

# 4. 传入新学科，生成完整提示词
formatted_prompt = few_shot_prompt.format(new_subject="LangChain")
print("少样本提示词：")
print(formatted_prompt)

# 5. 调用模型
result = chat_model.invoke([{"role": "user", "content": formatted_prompt}])
print("\n生成的LangChain学习方法：")
print(result.content)

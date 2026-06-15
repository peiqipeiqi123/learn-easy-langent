# 1. 导入需要的模块
import os 
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv

# 2. 加载API密钥
load_dotenv()

# 3. 配置 API Key
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")
if not BASE_URL:
    raise ValueError("未检测到 BASE_URL，请检查 .env 文件是否配置正确")

# 4. 初始化大模型（和LangChain案例一样）
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-v4-flash", # 注意：根据你使用的模型修改名称！！！！ 后面章节不再继续说明
    temperature=0.3
)

# 5. 定义 State
class WorkflowState(TypedDict, total=False):
    user_role: str  # 存储用户角色
    original_advice: str  # 存储原始学习建议
    simplified_advice: str  # 存储精简后的建议
    english_advice: str #存储翻译后的建议

# 6. 定义节点
def generate_advice(state: WorkflowState):
    prompt = f"给{state['user_role']}写一段50字左右的 AI 学习建议。"
    result = llm.invoke(prompt)
    return {"original_advice": result.content}

def simplify_advice(state: WorkflowState):
    prompt = f"把下面的学习建议精简到30字以内：{state['original_advice']}"
    result = llm.invoke(prompt)
    return {"simplified_advice": result.content}

def translate_to_english(state: WorkflowState):
      prompt = f"把这句话翻译成英文：{state['simplified_advice']}"
      result = llm.invoke(prompt)
      return {"english_advice": result.content}

# 7. 构建工作流
workflow = StateGraph(WorkflowState)

workflow.add_node("generate", generate_advice)
workflow.add_node("simplify", simplify_advice)
workflow.add_node("translate", translate_to_english) 

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "simplify")
workflow.add_edge("simplify", "translate")
workflow.add_edge("translate", END)

app = workflow.compile()

# 8. 执行
result = app.invoke({"user_role": "高校学生"})

# 9. 输出
print("原始学习建议：")
print(result["original_advice"])
print("\n精简后学习建议：")
print(result["simplified_advice"])
print("\n英文学习建议：")         
print(result["english_advice"])
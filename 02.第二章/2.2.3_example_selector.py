# ============================================================
# 2.2.3 工程化实践：ExampleSelector 动态示例选择
# 核心：根据用户输入的难度等级，自动筛选匹配的示例
# 依赖：learning_method_examples.json（同目录下）
# ============================================================
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import BaseExampleSelector
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
from typing import Dict, List

# 1. 环境初始化
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
    max_tokens=300
)

# 2. 从 JSON 文件加载示例（工程化：示例和代码分离，便于维护）
with open("learning_method_examples.json", "r", encoding="utf-8") as f:
    examples = json.load(f)


# 3. 自定义选择器：根据用户输入的 difficulty 筛选示例
class DifficultyExampleSelector(BaseExampleSelector):
    """按难度等级筛选匹配的示例"""

    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples

    def add_example(self, example: Dict[str, str]) -> None:
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[Dict]:
        target_difficulty = input_variables.get("difficulty", "easy")
        return [ex for ex in self.examples if ex.get("difficulty") == target_difficulty]


example_selector = DifficultyExampleSelector(examples=examples)

# 4. 构建少样本模板（用选择器替换固定示例列表）
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,  # 动态选择，不是固定列表
    example_prompt=PromptTemplate(
        input_variables=["subject", "difficulty", "method"],
        template="学科：{subject}\n难度：{difficulty}\n学习方法：{method}\n"
    ),
    example_separator="\n",
    prefix="少样本提示：",
    suffix="参考以上示例，回答：\n学科：{new_subject}\n难度：{difficulty}\n学习方法：",
    input_variables=["new_subject", "difficulty"]
)

# 5. 场景1：入门级（只匹配 difficulty=easy 的示例）
print("=" * 50)
print("入门级少样本提示词：")
print(formatted_prompt_easy := few_shot_prompt.format(new_subject="LangChain", difficulty="easy"))
result_easy = chat_model.invoke([{"role": "user", "content": formatted_prompt_easy}])
print("\n入门级学习方法：")
print(result_easy.content)

# 6. 场景2：进阶级（只匹配 difficulty=hard 的示例）
print("\n" + "=" * 50)
print("进阶级少样本提示词：")
print(formatted_prompt_hard := few_shot_prompt.format(new_subject="LangChain", difficulty="hard"))
result_hard = chat_model.invoke([{"role": "user", "content": formatted_prompt_hard}])
print("\n进阶级学习方法：")
print(result_hard.content)

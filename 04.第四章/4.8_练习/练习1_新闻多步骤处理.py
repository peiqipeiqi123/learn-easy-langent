# ============================================================
# 第四章 练习 1：新闻文本分类→提取核心事件→生成摘要
# 使用 RunnableSequence / | 运算符实现多步骤链式工作流
# ============================================================
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

# ============================================================
# Step 1: 新闻分类
# ============================================================
# 测试数据：
news_text = "6月16日，在世界杯小组赛夺冠热门西班牙 VS 佛得角的比赛中，佛得角40岁门将沃齐尼亚7次扑救，5万欧身价零封西班牙，最终以0-0逼平。赛后沃齐尼亚哭了“我这一生都在为这一刻努力”"
category_list = ["体育", "科技", "财经", "世界杯","娱乐"]

classify_prompt = PromptTemplate(
      input_variables=["news_text", "category_list"],
      template="""请将以下新闻分到最合适的类别中。
  类别列表：{category_list}
  新闻内容：{news_text}

  只输出类别名称，不要其他内容。"""
  )
classify_chain = classify_prompt | llm

# ============================================================
# Step 2: 提取核心事件
extract_prompt = PromptTemplate(
      input_variables=["news_text", "category"],
      template="""你是{category}新闻编辑，请从以下新闻中提取一个核心事件，用一句话概括。

  新闻内容：{news_text}

  核心事件："""
  )

extract_chain = extract_prompt | llm
# ============================================================


# ============================================================
# Step 3: 生成摘要
summary_prompt = PromptTemplate(
      input_variables=["category", "core_event"],
      template="""请根据以下{category}新闻的核心事件，写一段100字以内的新闻摘要。

  核心事件：{core_event}

  摘要："""
  )

summary_chain = summary_prompt | llm
# ============================================================


# ============================================================
# 串联 + 测试
# Step 1: 分类
cat_result = classify_chain.invoke({
      "news_text": news_text,
      "category_list": "、".join(category_list)
})
category = cat_result.content.strip()
print(f"【分类】{category}")

# Step 2: 提取核心事件
event_result = extract_chain.invoke({
      "news_text": news_text,
      "category": category
  })
core_event = event_result.content.strip()
print(f"【核心事件】{core_event}")

# Step 3: 生成摘要
summary_result = summary_chain.invoke({
      "category": category,
      "core_event": core_event
})
print(f"【摘要】{summary_result.content.strip()}")
# ============================================================

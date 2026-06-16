# ============================================================
# 第四章 练习 2：构建 PDF RAG 问答机器人（已完成）
#
# 练习要求 vs 实际实现：
#   ① 文档加载：PDF → Document 对象       → 4.4.1 + 4.4.3
#   ② 文本分割：RecursiveCharacterTextSplitter → 4.4.2
#   ③ 向量存储：嵌入模型 + 向量数据库      → Qwen3-Embedding + FAISS（教程用 Chroma，换了）
#   ④ 检索-生成：自定义提示词 + 链式调用   → 下方完整实现
#   ⑤ 评估调优：RAGAS 评估（选学）        → 暂未实现
#
# 注意：本文件从 4.3_rag/ 复制而来，路径已调整为相对路径
# ============================================================
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# ---------- 环境 ----------
load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
if not api_key:
    raise ValueError("未找到 API_KEY 环境变量，请检查 .env 文件配置")

# ---------- ① + ③ 嵌入模型（Qwen3-Embedding-0.6B，本地 CPU） ----------
# 路径说明：从 4.8_练习/ 上一级到 4.3_rag/models/Qwen
embedding_model_path = "../4.3_rag/models/Qwen"
if not os.path.exists(embedding_model_path):
    raise FileNotFoundError(f"嵌入模型路径不存在：{embedding_model_path}")

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ---------- ③ 加载 FAISS 向量库（4.4.3 构建） ----------
vector_db = FAISS.load_local(
    folder_path="../4.3_rag/faiss_db",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
    index_name="local_cpu_faiss_index"
)

# ---------- ④ 检索器 ----------
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
)

# ---------- ④ 大模型 ----------
llm = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model="deepseek-chat",
    temperature=0.3,
    timeout=30,
    max_retries=2
)

# ---------- ② + ④ 提示词 + 检索-生成链 ----------
def format_docs(docs):
    """将检索到的文档片段合并为统一文本"""
    return "\n\n".join([doc.page_content for doc in docs])

system_prompt = """你是一个专业的RAG系统问答助手，必须基于以下提供的参考资料（context）回答用户问题。
规则：
1. 答案必须严格基于参考资料，不能编造未提及的信息；
2. 语言简洁明了，分点说明（如果有多个要点）；
3. 每个要点搭配1个简单案例，帮助理解。

参考资料：{context}"""

custom_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# 完整 RAG 链：检索 → 格式化 → 拼提示词 → LLM → 字符串输出
rag_qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_prompt
    | llm
    | StrOutputParser()
)

# ---------- ⑤ 测试 + 评估 ----------
# 10 个问题覆盖论文核心知识点：harness定义、scaffold区别、agent关系、
# skills/tools、产品案例、训练推理、企业应用等
test_questions = [
    # === 定义类 ===
    "什么是harness？它和scaffold有什么区别？",
    "harness engineering的核心价值是什么？",
    "scaffold在AI Agent系统中扮演什么角色？",

    # === 对比类 ===
    "harness和直接使用大模型相比，优势在哪里？",
    "工具（tool）和技能（skill）在harness中有什么区别？",

    # === 应用类 ===
    "企业内部知识库问答为什么适合用harness？",
    "Claude Code、Codex、Cursor这些产品是如何使用harness的？",

    # === 工程类 ===
    "训练时的harness和推理时的harness有什么不同？",
    "构建一个好的Agent Harness需要关注哪些关键设计决策？",
    "harness如何影响Agent与用户的交互方式？",
]

# 评估说明（人工复核）：
#   ① 检索相关度：看「参考资料」片段是否与问题相关
#   ② 事实一致性：看答案是否基于片段，有无编造
#   ③ 覆盖率：10 道题是否都得到了有据可依的回答

for i, question in enumerate(test_questions):
    print(f"\n{'='*60}")
    print(f"测试问题{i+1}/10：{question}")
    print("="*60)

    answer = rag_qa_chain.invoke(question)
    print("\n生成答案：")
    print(answer)

    print("\n参考资料（检索片段）：")
    sources = retriever.invoke(question)
    for j, doc in enumerate(sources):
        print(f"\n参考片段{j+1}：{doc.page_content[:150]}...")
        print(f"来源页码：{doc.metadata.get('page', '未知')}")

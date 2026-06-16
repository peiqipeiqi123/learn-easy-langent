# ============================================================
# 第四章 练习 3：错误处理 + 分支降级机制
#
# 架构：
#   核心链：deepseek-v4-flash + Qwen3-Embedding（本地）
#   降级链：deepseek-v4-pro  + Qwen3-Embedding（共用嵌入）
#   触发条件：API 超时、密钥错误、检索无结果 → 自动降级
#
# [自改] 教程要求用 GPT-4 → GPT-3.5 降级，我换成 DeepSeek 模型
# ============================================================
import os
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# ============================================================
# 共享资源：嵌入模型 + 向量库（核心链和降级链共用）
# ============================================================
embedding_model_path = "../4.3_rag/models/Qwen"
if not os.path.exists(embedding_model_path):
    raise FileNotFoundError(f"嵌入模型路径不存在：{embedding_model_path}")

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ============================================================
# 定义两条链
# ============================================================

def build_rag_chain(llm):
    """用给定的 LLM 构建一条 RAG 链"""
    try:
        vector_db = FAISS.load_local(
            folder_path="../4.3_rag/faiss_db",
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
            index_name="local_cpu_faiss_index"
        )
    except Exception as e:
        # 向量库加载失败 → 友好提示，不崩溃
        raise RuntimeError(f"[向量库加载失败] {e}\n请确认 faiss_db 已通过 4.4.3 构建。")

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的RAG问答助手，基于以下参考资料回答。\n参考资料：{context}"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def retry_invoke(chain, question, max_retries=3):
    """带重试的 invoke：失败自动重试，间隔递增"""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(question)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                print(f"  [重试 {attempt}/{max_retries}] {e}")
                time.sleep(1 * attempt)  # 递增等待
    raise last_error


# ============================================================
# 初始化两条链
# ============================================================
print("初始化核心链（deepseek-v4-flash）...")
core_llm = ChatOpenAI(
    api_key=API_KEY, base_url=BASE_URL,
    model="deepseek-v4-flash",
    temperature=0.3, timeout=15, max_retries=1
)
core_chain = build_rag_chain(core_llm)

print("初始化降级链（deepseek-v4-pro）...")
fallback_llm = ChatOpenAI(
    api_key=API_KEY, base_url=BASE_URL,
    model="deepseek-v4-pro",
    temperature=0.3, timeout=30, max_retries=2
)
fallback_chain = build_rag_chain(fallback_llm)


# ============================================================
# 带降级的问答函数
# ============================================================
def safe_ask(question: str) -> str:
    """核心链优先，失败则自动降级"""

    # 1. 输入校验
    if not question or not question.strip():
        return "[输入错误] 问题不能为空。"

    # 2. 尝试核心链
    try:
        answer = retry_invoke(core_chain, question, max_retries=2)
        print("  [使用 核心链 回答]")
        return answer
    except Exception as e:
        print(f"  [核心链失败] {e}")

    # 3. 降级
    try:
        print("  [降级到备用链]")
        answer = retry_invoke(fallback_chain, question, max_retries=3)
        print("  [使用 降级链 回答]")
        return answer
    except Exception as e:
        # 4. 全部失败 → 友好降级回复
        return f"[服务暂时不可用] 核心链和备用链均失败：{e}。请稍后重试或检查 API 密钥。"


# ============================================================
# 测试：覆盖正常场景 + 异常场景
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("场景 1：正常问答（核心链）")
    print("=" * 60)
    print(safe_ask("harness engineering 的核心价值是什么？"))

    print("\n" + "=" * 60)
    print("场景 2：空输入（触发输入校验）")
    print("=" * 60)
    print(safe_ask(""))

    print("\n" + "=" * 60)
    print("场景 3：正常问答（验证降级链可用）")
    print("=" * 60)
    print(safe_ask("什么是 scaffold？"))

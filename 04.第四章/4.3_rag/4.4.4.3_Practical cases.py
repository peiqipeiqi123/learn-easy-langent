# [自改] 只保留相似性检索 k=3（单篇论文最优），其余 3 种检索器注释掉
# 模型路径改为 ./models/Qwen
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

embedding_model_name = "./models/Qwen"

# 初始化本地CPU运行的嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={
        "device": "cpu",  # 强制CPU运行，无需GPU
        # 如需加载量化模型，可添加以下配置（按需）
        # "trust_remote_code": True,
        # "load_in_8bit": False
    },
    encode_kwargs={
        "normalize_embeddings": True  # 归一化向量，提升检索效果
    }
)

# 加载已有的FAISS向量数据库
vector_db = FAISS.load_local(
    folder_path="./faiss_db",  # 之前存储向量的路径
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
    index_name="local_cpu_faiss_index"
)
print("向量数据库加载成功！")

# 相似性检索（k=3）：单篇论文场景最优
retriever_similar_k3 = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# # 备选：相似性检索（k=5）—— 文档内容跨越大时可用
# retriever_similar_k5 = vector_db.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 5}
# )

# # 备选：MMR检索（偏向相关性 λ=0.8）
# retriever_mmr_high_rel = vector_db.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.8}
# )

# # 备选：MMR检索（偏向多样性 λ=0.3）
# retriever_mmr_high_div = vector_db.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.3}
# )

# 5. 定义测试查询并执行检索（适配v0.1+ invoke方法）
test_query = "harness系统的核心价值是什么？"

def test_retriever(retriever: BaseRetriever, retriever_name: str):
    """测试检索器并打印结果"""
    try:
        # 核心适配：使用invoke()替代旧的get_relevant_documents()
        results: list[Document] = retriever.invoke(test_query)
        print(f"=== {retriever_name} 检索结果（共{len(results)}条） ===")
        for i, doc in enumerate(results):
            print(f"\n[{i+1}] 内容：{doc.page_content[:120]}...")
            print(f"   来源：{doc.metadata.get('source', '未知')}")
        print("\n" + "-"*80 + "\n")
    except Exception as e:
        raise RuntimeError(f"{retriever_name} 检索失败：{str(e)}")

# 执行检索
test_retriever(retriever_similar_k3, "相似性检索（k=3）")

# 可选：补充相似性评分展示（FAISS特有）
print("=== 相似性检索（k=3）带评分结果 ===")
docs_with_scores = vector_db.similarity_search_with_score(test_query, k=3)
for i, (doc, score) in enumerate(docs_with_scores):
    print(f"\n[{i+1}] 评分：{round(score, 4)} | 内容：{doc.page_content[:80]}...")
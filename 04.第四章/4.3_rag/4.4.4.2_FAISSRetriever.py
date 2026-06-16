# [自改] 模型路径改为 ./models/Qwen（教程里是指向 Qwen3-Embedding-0.6B 子目录）
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# 本地Qwen嵌入模型路径
embedding_model_name = "./models/Qwen"

# 初始化本地CPU运行的嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={
        "device": "cpu"  # 强制使用CPU运行
    },
    encode_kwargs={
        "normalize_embeddings": True  # 归一化向量，提升检索效果
    }
)

# 加载已有的FAISS向量数据库
try:
    vector_db = FAISS.load_local(
        folder_path="./faiss_db",  # 之前的持久化路径
        embeddings=embeddings,
        allow_dangerous_deserialization=True, 
        index_name="local_cpu_faiss_index"
    )
    print("FAISS向量库加载成功！")
except FileNotFoundError:
    raise FileNotFoundError("未找到 ./faiss_db 文件夹，请确认向量库已正确保存")
except Exception as e:
    raise RuntimeError(f"加载FAISS向量库失败：{str(e)}")

# 创建检索器（v0.1+ 规范）
retriever: BaseRetriever = vector_db.as_retriever(
    search_kwargs={"k": 3},  # 每次检索返回3个最相关的片段
    # 可选：按分数阈值检索（按需启用）
    # search_type="similarity_score_threshold",
    # search_kwargs={"k": 3, "score_threshold": 0.5}
)

# 测试检索
query = "LangChain的SequentialChain有什么用？"
try:
    # 核心修改：替换 get_relevant_documents → invoke（Runnable接口标准）
    retrieved_docs: list[Document] = retriever.invoke(query)

    print(f"\n检索到的相关片段（{len(retrieved_docs)}个）：")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n片段{i+1}：")
        print(f"内容：{doc.page_content}")
        print(f"来源文件：{doc.metadata.get('source', '未知')}")

    # 如需获取检索评分（补充完整可运行的评分获取逻辑）
    print("\n===== 带评分的检索结果 =====")
    docs_with_scores = vector_db.similarity_search_with_score(query, k=3)
    for i, (doc, score) in enumerate(docs_with_scores):
        print(f"\n片段{i+1}（相关性评分：{round(score, 4)}）：")
        print(f"内容：{doc.page_content}")
        
except Exception as e:
    raise RuntimeError(f"检索向量库失败：{str(e)}")
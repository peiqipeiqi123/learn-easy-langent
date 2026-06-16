# [自改] 教程用 test.txt + 默认嵌入模型路径
# 我改成：PDF 文件 + 本地 Qwen3-Embedding-0.6B 模型（路径 ./models/Qwen）
# 参考：https://easy-langent.datawhale.cc/guide/chapter4.html
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

pdf_path = os.path.join("knowledge_base", "Harness, Scaffold, and the AI Agent Terms Worth Getting Right.pdf")
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"文档文件不存在：{pdf_path}")

# 加载文本文档
loader = PDFPlumberLoader(str(pdf_path))
txt_docs: list[Document] = loader.load()

# 文本分割（使用最新的 RecursiveCharacterTextSplitter 配置）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,          # 每个文本块的大小
    chunk_overlap=50,        # 块之间的重叠长度（提升上下文连续性）
    length_function=len,     # 长度计算函数（中文用len即可）
    is_separator_regex=False # 显式指定非正则分隔符（默认值，增加可读性）
)
split_docs: list[Document] = text_splitter.split_documents(txt_docs)
print(f"分割后的文本片段数：{len(split_docs)}")

# 3. 初始化本地CPU运行的嵌入模型（替换QwenEmbeddings）
embedding_model_name = "./models/Qwen"

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={
        "device": "cpu"  # 强制使用CPU运行，无需GPU
    },
    encode_kwargs={
        "normalize_embeddings": True  # 归一化向量，提升检索效果
    }
)

# 4. 构建并持久化FAISS向量库
try:
    # 生成向量并初始化FAISS（本地CPU计算，首次运行会下载模型，需联网）
    vector_db = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings,
    )

    # 持久化向量库到本地
    vector_db.save_local(
        folder_path="./faiss_db",
        index_name="local_cpu_faiss_index"  # 索引名改为本地CPU版
    )
    print("向量存储完成！向量数据已保存到 ./faiss_db 文件夹")
except Exception as e:
    raise RuntimeError(f"构建/保存向量库失败：{str(e)}")

# 5. 相似性检索测试
query = "LangChain的链式工作流有哪些类型？"
try:
    # 一次性获取带评分的检索结果
    retrieved_docs_with_scores = vector_db.similarity_search_with_score(query, k=3)
    
    print(f"\n与问题「{query}」最相关的3个文本片段：")
    for i, (doc, score) in enumerate(retrieved_docs_with_scores):
        print(f"\n片段{i+1}：")
        print(f"内容：{doc.page_content}")
        print(f"相关性评分（越小越相似）：{round(score, 4)}")
        print(f"来源：{doc.metadata.get('source', '未知')}")
except Exception as e:
    raise RuntimeError(f"检索向量库失败：{str(e)}")
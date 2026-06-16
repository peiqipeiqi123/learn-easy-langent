# [自改] 教程用 TextLoader 加载 .txt，我的知识文档是 PDF → 换成 PDFPlumberLoader
# 文件名也改成了自己的 PDF 路径
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader   # LangChain推荐的基础文本加载器
from pathlib import Path  # 官方推荐用pathlib处理路径（比os.path更现代）

# 1. 加载文档（推荐用Path处理路径，避免跨系统兼容问题）
pdf_path = Path("knowledge_base") / "Harness, Scaffold, and the AI Agent Terms Worth Getting Right.pdf"
loader = PDFPlumberLoader(str(pdf_path))
txt_docs = loader.load()  # 返回Document对象列表（含内容+元数据）

# 2. 初始化分割器（LangChain推荐参数）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,          # 中文片段推荐长度：200-500字
    chunk_overlap=50,        # 重叠长度：建议为chunk_size的10%-20%，避免跨片段语义丢失
    length_function=len,     # 中文用len计数，英文可改用tiktoken.count_tokens
    separators=["\n\n", "\n", "。", "！", "？", "，", "；", "、"]  # 中文推荐分隔符优先级
)

# 3. 执行分割（split_documents为官方推荐方法，接收Document列表）
split_docs = text_splitter.split_documents(txt_docs)

# 4. 验证结果
print(f"原始文档数：{len(txt_docs)}")
print(f"分割后片段数：{len(split_docs)}")
print("\n前3个片段示例：")
for i, doc in enumerate(split_docs[:3]):
    print(f"\n片段{i+1}（字符数：{len(doc.page_content)}）：")
    print(doc.page_content.strip())
    print(f"片段元数据：{doc.metadata}")  # 保留原始文档路径等元数据（检索时有用）
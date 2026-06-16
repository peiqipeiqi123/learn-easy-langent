# [自改] 教程用 TextLoader，我改成批量多格式加载（PDF/TXT/DOCX/MD）
# 实际只用 PDF 做测试，放在 knowledge_base/ 目录下
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader
)
import os

def batch_load_documents(folder_path):
    """
    批量加载文件夹内的所有官方支持格式文档（基于新版加载器）
    :param folder_path: 知识库文件夹路径
    :return: 所有文档的Document对象列表
    """
    all_docs = []
    # 遍历文件夹内所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 跳过文件夹，只处理文件
        if os.path.isdir(file_path):
            continue
        # 根据文件后缀选择对应的官方推荐加载器
        try:
            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)  # 基础款，复杂场景可替换为PDFPlumberLoader
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif filename.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                print(f"不支持的文件格式：{filename}")
                continue
            # 加载并添加到文档列表
            docs = loader.load()
            all_docs.extend(docs)
            print(f"成功加载：{filename}，生成{len(docs)}个Document对象")
        except Exception as e:
            print(f"加载失败：{filename}，错误信息：{str(e)}")
    return all_docs

# 测试批量加载
if __name__ == "__main__":
    knowledge_base_path = "knowledge_base"
    # 确保知识库文件夹存在
    if not os.path.exists(knowledge_base_path):
        os.makedirs(knowledge_base_path)
        print(f"已自动创建知识库文件夹：{knowledge_base_path}，请放入测试文档")
    else:
        all_docs = batch_load_documents(knowledge_base_path)
        print(f"\n批量加载完成，总Document对象数：{len(all_docs)}")
        # 打印每个文档的基本信息
        for i, doc in enumerate(all_docs):
            print(f"\n文档{i+1}：")
            print(f"内容预览：{doc.page_content[:100]}...")
            print(f"元数据：{doc.metadata}")
"""
第五章 学习助手 Agent — 工具模块（9 个 Tool）

Tool 设计原则：
- 每个 @tool 的 docstring 就是 LLM 的"使用说明书"，需清晰描述何时调用、参数含义
- 所有 Tool 统一返回 str，便于 LLM 理解和后续处理
- 错误不崩溃，优雅返回错误信息给 LLM
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

from langchain_core.tools import tool
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from config import (
    get_llm, get_embeddings, KNOWLEDGE_BASE_DIR, SESSIONS_DIR,
    DEFAULT_SESSION, CHAPTER5_DIR, PROJECT_ROOT,
)
from memory import get_memory
from prompts import (
    SUMMARIZE_PROMPT,
    COMPARE_PROMPT,
    RELATED_WORK_PROMPT,
    FLOWCHART_PROMPT,
)


# =====================
# 全局状态（模块级，供所有 Tool 共享）
# =====================

# 当前会话名
_current_session: str = DEFAULT_SESSION

# 存储已加载文档的全部 chunk（doc_id → list[Document]）
# 用于 remove_document 时重建 FAISS 索引
_all_chunks: dict[str, list[Document]] = {}

# FAISS 向量数据库实例（懒初始化）
_vector_db: FAISS | None = None

# 文本分割器（全局复用）
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    length_function=len,
)


# =====================
# 会话路径管理
# =====================

def _session_dir(name: str | None = None) -> Path:
    """获取指定会话的存储目录。"""
    name = name or _current_session
    d = SESSIONS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _faiss_dir(name: str | None = None) -> Path:
    """获取指定会话的 FAISS 索引目录。

    FAISS C++ 层在 Windows 上不支持 Unicode 路径，
    因此用 hash 代替原始会话名作为目录名。
    """
    from config import PROJECT_ROOT
    name = name or _current_session
    safe = hashlib.md5(name.encode()).hexdigest()[:8]
    d = PROJECT_ROOT / "faiss_sessions" / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_session(name: str) -> None:
    """切换到指定会话。"""
    global _current_session, _vector_db
    _current_session = name
    _vector_db = None  # 强制重新加载


def get_session() -> str:
    """获取当前会话名。"""
    return _current_session


def rename_faiss(old_name: str, new_name: str) -> None:
    """重命名会话时同步搬移 FAISS 索引。"""
    old_d = _faiss_dir(old_name)
    new_d = _faiss_dir(new_name)
    if old_d.exists() and not new_d.exists():
        import shutil
        shutil.copytree(str(old_d), str(new_d))
        shutil.rmtree(str(old_d))


# =====================
# 内部辅助函数
# =====================

def _safe_invoke(chain, **kwargs) -> str:
    """
    安全调用 ChatPromptTemplate chain，自动转义参数中的 {} 防止模板注入。

    ★ Insight ─────────────────────────────────────
    ChatPromptTemplate 把 {xxx} 当变量占位符，但用户输入和文档内容
    随时可能包含 {}（JOSN、代码段等）。这个包装函数在每次 invoke 前
    把所有参数值里的 { 和 } 转义为 {{ 和 }}，避免误解析。
    ─────────────────────────────────────────────────
    """
    safe_kwargs = {
        k: v.replace("{", "{{").replace("}", "}}") if isinstance(v, str) else v
        for k, v in kwargs.items()
    }
    return chain.invoke(safe_kwargs)


def _detect_file_type(file_path: str) -> str | None:
    """
    通过文件头魔数检测真实文件类型，不依赖扩展名。
    返回标准扩展名（如 .pdf），无法识别则返回 None。

    ★ Insight ─────────────────────────────────────
    魔数（Magic Number）是文件开头的几个字节，标识文件真实格式。
    比如 PDF 永远是 %PDF 开头，改名改扩展名不影响。
    ─────────────────────────────────────────────────
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
    except Exception:
        return None

    if header[:4] == b"%PDF":
        return ".pdf"
    if header[:4] == b"PK\x03\x04":
        # ZIP 格式 → 可能是 DOCX（进一步检查内部文件）
        return ".docx"
    # 纯文本文件没有固定魔数，回退到扩展名判断
    return None


def _make_doc_id(file_path: str) -> str:
    """根据文件路径生成唯一文档 ID（用于索引和查找）。"""
    # 用文件名的 hash 前缀 + 文件名，方便人类识别
    name = Path(file_path).name
    suffix = hashlib.md5(file_path.encode()).hexdigest()[:6]
    return f"{name}_{suffix}"


def _rebuild_faiss() -> FAISS:
    """用 _all_chunks 中的所有 chunk 重建 FAISS 索引，存入当前会话目录。"""
    global _vector_db
    embeddings = get_embeddings()
    all_docs = []
    for chunks in _all_chunks.values():
        all_docs.extend(chunks)
    faiss_dir = str(_faiss_dir())
    if not all_docs:
        placeholder = Document(page_content="（知识库为空，请先加载文档）", metadata={"source": "system"})
        _vector_db = FAISS.from_documents([placeholder], embeddings)
    else:
        _vector_db = FAISS.from_documents(all_docs, embeddings)
    _vector_db.save_local(faiss_dir, index_name="learning_assistant")
    return _vector_db


def _chunks_file(name: str | None = None) -> Path:
    return _session_dir(name) / "chunks.json"


def save_chunks() -> None:
    """保存当前会话的 chunk 数据到磁盘。"""
    data = {}
    for doc_id, doc_list in _all_chunks.items():
        data[doc_id] = [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in doc_list
        ]
    _chunks_file().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(session_name: str | None = None) -> bool:
    """从磁盘恢复指定会话的 chunk 数据。返回是否成功。"""
    path = _chunks_file(session_name)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _all_chunks.clear()
        for doc_id, doc_list in data.items():
            _all_chunks[doc_id] = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in doc_list
            ]
        return True
    except Exception:
        return False


def _get_vector_db() -> FAISS:
    """获取或重建当前会话的 FAISS 向量数据库。"""
    global _vector_db
    if _vector_db is not None:
        return _vector_db
    embeddings = get_embeddings()
    faiss_dir = str(_faiss_dir())
    index_path = Path(faiss_dir) / "learning_assistant.faiss"
    if index_path.exists():
        try:
            _vector_db = FAISS.load_local(
                faiss_dir,
                embeddings,
                allow_dangerous_deserialization=True,
                index_name="learning_assistant",
            )
            return _vector_db
        except Exception:
            pass
    return _rebuild_faiss()


# =====================
# Tool 1: 加载文档
# =====================

# 支持的文件扩展名
_SUPPORTED_EXTS = (".pdf", ".txt", ".md", ".docx")

# 扩展名 → Loader 工厂
def _make_loader(file_path: str, suffix: str):
    """根据扩展名返回对应的 LangChain Loader 实例。"""
    if suffix == ".pdf":
        return PDFPlumberLoader(file_path)
    elif suffix == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    elif suffix == ".md":
        return UnstructuredMarkdownLoader(file_path)
    elif suffix == ".docx":
        return Docx2txtLoader(file_path)
    raise ValueError(f"Unsupported format: {suffix}")


def _load_one_file(file_path: Path) -> tuple[str, list[Document]]:
    """
    加载单个文件并完成分块、元数据标记、文档注册。
    返回 (doc_id, chunks)，不触发 FAISS 重建（由调用方统一处理）。
    """
    path = file_path.resolve()
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_EXTS:
        detected = _detect_file_type(str(path))
        suffix = detected if detected else suffix
    if suffix not in _SUPPORTED_EXTS:
        raise ValueError(f"不支持的文件格式：{suffix}")

    loader = _make_loader(str(path), suffix)
    raw_docs: list[Document] = loader.load()
    chunks = _text_splitter.split_documents(raw_docs)

    doc_id = _make_doc_id(str(path))
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["source"] = path.name
        chunk.metadata["page"] = chunk.metadata.get("page", "?")

    _all_chunks[doc_id] = chunks
    memory = get_memory()
    memory.docs.register(
        doc_id=doc_id, name=path.name, path=str(path), num_chunks=len(chunks)
    )
    return doc_id, chunks


@tool
def add_document(file_path: str) -> str:
    """加载一个文档文件（PDF/TXT/DOCX/MD）或文件夹到知识库中。
    调用时机：用户要求"加载""添加""导入"文档时。
    参数：file_path - 文档路径或文件夹路径。"""
    path = Path(file_path)
    # 路径解析
    if not path.is_absolute():
        for candidate in (KNOWLEDGE_BASE_DIR / path, CHAPTER5_DIR.parent / path):
            if candidate.exists():
                path = candidate
                break
        else:
            return (
                f"❌ 文件不存在：{file_path}\n"
                f"已搜索：{KNOWLEDGE_BASE_DIR / path}、{CHAPTER5_DIR.parent / path}"
            )

    # ---- 文件夹：批量加载 ----
    if path.is_dir():
        results, errors = [], []
        for f in sorted(path.iterdir()):
            if not f.is_file() or f.suffix.lower() not in _SUPPORTED_EXTS:
                continue
            try:
                doc_id, chunks = _load_one_file(f)
                sample = "\n\n".join([c.page_content for c in chunks[:3]])
                try:
                    memory = get_memory()
                    memory.update_document_summary(doc_id, sample, get_llm())
                except Exception:
                    pass
                results.append(f"  ✅ {f.name}（{len(chunks)}块）")
            except Exception as e:
                errors.append(f"  ❌ {f.name}: {e}")
        if results:
            _rebuild_faiss()
        summary = f"📁 文件夹批量加载完成：\n成功 {len(results)} 个\n"
        if results:
            summary += "\n".join(results[-20:])
        if errors:
            summary += f"\n失败 {len(errors)} 个：\n" + "\n".join(errors[-5:])
        return summary

    # ---- 单文件加载 ----
    try:
        doc_id, chunks = _load_one_file(path)
    except Exception as e:
        return f"❌ 加载文档失败：{e}"

    _rebuild_faiss()

    # 生成摘要
    sample_text = "\n\n".join([chunk.page_content for chunk in chunks[:3]])
    summary_display = ""
    try:
        summary = get_memory().update_document_summary(doc_id, sample_text, get_llm())
        summary_display = "\n\n📝 自动生成摘要：\n" + summary
    except Exception:
        pass

    return (
        f"✅ 文档加载成功！\n"
        f"- 文件名：{path.name}\n- 文档ID：`{doc_id}`\n"
        f"- 分块数：{len(chunks)}（500字/块，重叠80字）\n- 已加入检索索引"
        f"{summary_display}"
    )


# =====================
# Tool 2: 列出文档
# =====================

@tool
def list_documents() -> str:
    """列出当前知识库中所有已加载的文档及其摘要。
    调用时机：用户询问"有哪些文档""已加载了什么""看看文档列表"时。"""
    memory = get_memory()
    return "📋 **已加载文档列表**\n" + memory.docs.to_display()


# =====================
# Tool 3: 移除文档
# =====================

@tool
def remove_document(doc_id: str) -> str:
    """从知识库中移除指定文档。
    调用时机：用户说"移除""删除""不要"某个文档时。
    参数：doc_id - 文档ID（由 add_document 生成，或通过 list_documents 查看）。"""
    memory = get_memory()
    if doc_id not in memory.docs:
        available = ", ".join([d["id"] for d in memory.docs.list_all()]) or "无"
        return f"❌ 未找到文档 `{doc_id}`。当前可用文档ID：{available}"

    doc_info = memory.docs.remove(doc_id)
    _all_chunks.pop(doc_id, None)
    _rebuild_faiss()

    return f"✅ 已移除文档：{doc_info['name']}（ID: {doc_id}）"


# =====================
# Tool 4: 检索问答
# =====================

@tool
def search_documents(query: str) -> str:
    """在已加载的所有文档中搜索相关内容，用于回答用户的具体问题。
    调用时机：用户提出的问题需要基于文档内容回答时。
    参数：query - 用户的搜索查询（自然语言问题）。"""
    memory = get_memory()
    if len(memory.docs) == 0:
        return "⚠️ 知识库为空，无法检索。请先使用 add_document 加载文档。"

    try:
        vector_db = _get_vector_db()
        # 检索最相关的 5 个片段（带相关性分数）
        results = vector_db.similarity_search_with_score(query, k=5)

        if not results:
            return "🔍 未在已加载文档中找到相关内容。"

        # 格式化检索结果
        formatted = []
        seen_content = set()
        for doc, score in results:
            content = doc.page_content.strip()
            # 去重（FAISS 可能返回重复内容）
            if content in seen_content:
                continue
            seen_content.add(content)

            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "?")
            relevance = max(0, min(1, 1 - score))  # 将 FAISS L2 距离转为相似度
            formatted.append(
                f"---\n📄 来源: {source} (页码: {page}) | 相关度: {relevance:.0%}\n\n{content}"
            )

        context = "\n".join(formatted)

        # 调用 LLM 基于检索结果生成答案
        from langchain_core.prompts import ChatPromptTemplate

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个学术问答助手。请基于以下检索到的文档片段回答用户问题。

规则：
1. 答案必须基于提供的文档片段，不可编造
2. 如果文档片段不足以回答，请诚实说明
3. 使用 Markdown 格式作答，分点清晰
4. 在答案中标注信息来源（文档名 + 页码）

检索到的文档片段：
{context}"""),
            ("human", "用户问题：{query}\n请基于以上文档片段回答："),
        ])

        chain = qa_prompt | get_llm() | StrOutputParser()
        answer = _safe_invoke(chain, context=context, query=query)

        return f"{answer}\n\n---\n📚 检索来源：共 {len(formatted)} 个相关片段"

    except Exception as e:
        return f"❌ 检索失败：{e}"


# =====================
# Tool 5: 文档摘要
# =====================

@tool
def summarize_document(doc_id: str) -> str:
    """为知识库中的指定文档生成结构化摘要（包含问题、方法、关键发现、贡献、关键词）。
    调用时机：用户说"总结""摘要""概括"某篇文档时。
    参数：doc_id - 文档ID（通过 list_documents 查看可用文档）。"""
    memory = get_memory()
    if doc_id not in memory.docs:
        available = ", ".join([d["id"] for d in memory.docs.list_all()]) or "无"
        return f"❌ 未找到文档 `{doc_id}`。可用ID：{available}"

    doc_info = memory.docs.get(doc_id)
    chunks = _all_chunks.get(doc_id, [])
    if not chunks:
        return f"❌ 文档 `{doc_id}` 的内容数据丢失，请重新加载。"

    # 拼接文档内容（限制长度避免爆 Token）
    content = "\n\n".join([chunk.page_content for chunk in chunks[:6]])

    chain = SUMMARIZE_PROMPT | get_llm() | StrOutputParser()
    result = _safe_invoke(chain, doc_name=doc_info["name"], content=content)

    # 尝试解析 JSON
    try:
        parsed = json.loads(result.strip())
        formatted = json.dumps(parsed, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        formatted = result.strip()

    # 更新文档摘要到记忆
    memory.docs.update_summary(doc_id, formatted)

    return f"📝 **{doc_info['name']}** 结构化摘要：\n\n```json\n{formatted}\n```"


# =====================
# Tool 6: 文档对比
# =====================

@tool
def compare_documents(doc1_id: str, doc2_id: str) -> str:
    """对比知识库中的两篇文档，分析其研究问题、方法、实验、结论的异同，并生成对比 Mermaid 图。
    调用时机：用户说"比较""对比""区别"两篇文档时。
    参数：doc1_id - 第一篇文档ID；doc2_id - 第二篇文档ID。"""
    memory = get_memory()
    for did in [doc1_id, doc2_id]:
        if did not in memory.docs:
            available = ", ".join([d["id"] for d in memory.docs.list_all()]) or "无"
            return f"❌ 未找到文档 `{did}`。可用ID：{available}"

    doc1 = memory.docs.get(doc1_id)
    doc2 = memory.docs.get(doc2_id)

    chunks1 = _all_chunks.get(doc1_id, [])
    chunks2 = _all_chunks.get(doc2_id, [])
    content1 = "\n\n".join([c.page_content for c in chunks1[:4]])
    content2 = "\n\n".join([c.page_content for c in chunks2[:4]])

    chain = COMPARE_PROMPT | get_llm() | StrOutputParser()
    result = _safe_invoke(
        chain,
        doc1_name=doc1["name"],
        doc1_summary=doc1["summary"] or "暂无摘要",
        doc1_content=content1,
        doc2_name=doc2["name"],
        doc2_summary=doc2["summary"] or "暂无摘要",
        doc2_content=content2,
    )

    # 尝试解析 JSON，提取 Mermaid 图表
    try:
        parsed = json.loads(result.strip())
        mermaid = parsed.get("mermaid_diagram", "")
        output = json.dumps(
            {k: v for k, v in parsed.items() if k != "mermaid_diagram"},
            ensure_ascii=False, indent=2,
        )
        if mermaid:
            output += f"\n\n### 关系图\n```mermaid\n{mermaid}\n```"
    except json.JSONDecodeError:
        output = result.strip()

    return f"🔬 **{doc1['name']}** vs **{doc2['name']}** 对比分析：\n\n{output}"


# =====================
# Tool 7: 写 Related Work
# =====================

@tool
def write_related_work(topic: str) -> str:
    """基于知识库中已加载的所有文献，撰写一段学术风格的 Related Work（相关工作）段落。
    调用时机：用户要求"写related work""写相关工作""帮我写文献综述段落"时。
    参数：topic - 用户的研究主题/论文题目。"""
    memory = get_memory()
    if len(memory.docs) == 0:
        return "⚠️ 知识库为空。请先加载相关文献后再请求撰写 Related Work。"

    # 收集所有文档的信息
    docs_info_parts = []
    for doc_id, info in memory.docs.items():
        chunks = _all_chunks.get(doc_id, [])
        sample = "\n".join([c.page_content[:300] for c in chunks[:3]])
        docs_info_parts.append(
            f"### {info['name']}\n摘要：{info['summary'] or '无'}\n内容片段：{sample[:600]}"
        )
    docs_info = "\n\n".join(docs_info_parts)

    chain = RELATED_WORK_PROMPT | get_llm() | StrOutputParser()
    result = _safe_invoke(chain, topic=topic, documents_info=docs_info)

    return f"✍️ 基于 {len(memory.docs)} 篇文献生成的 Related Work：\n\n{result}"


# =====================
# Tool 8: 联网搜索
# =====================

@tool
def web_search(query: str) -> str:
    """联网搜索最新信息，获取标题、摘要和来源链接。
    调用时机：用户问的问题无法从已加载文档中找到，或明确要求"搜索""查一下""最新"时。
    参数：query - 搜索查询词。"""
    from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=5))

        if not raw:
            return f"🔍 未找到与「{query}」相关的搜索结果。"

        formatted = []
        for i, r in enumerate(raw, 1):
            title = r.get("title", "无标题")
            body = r.get("body", "无摘要")
            href = r.get("href", "")
            formatted.append(f"{i}. **{title}**\n   {body}\n   🔗 {href}")

        from langchain_core.prompts import ChatPromptTemplate

        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """请基于以下联网搜索结果，用中文总结和回答用户的问题。
规则：
1. 整合多个搜索结果，给出有信息量的回答
2. 标注信息来源编号
3. 如果搜索结果不足以回答，请诚实说明
4. 使用 Markdown 格式"""),
            ("human", "用户问题：{query}\n\n搜索结果：\n{results}\n\n请回答："),
        ])

        chain = summary_prompt | get_llm() | StrOutputParser()
        answer = _safe_invoke(chain, query=query, results="\n".join(formatted))

        return f"🌐 联网搜索结果：\n\n{answer}\n\n---\n📎 原始搜索：\n" + "\n".join(formatted)

    except Exception as e:
        return f"❌ 联网搜索失败：{e}"
    except Exception as e:
        return f"❌ 联网搜索失败：{e}"


# =====================
# Tool 9: 生成流程图
# =====================

@tool
def generate_flowchart(description: str) -> str:
    """根据用户描述，生成 Mermaid 格式的流程图或关系图。
    调用时机：用户要求"画图""流程图""结构图""关系图""思维导图""可视化"时。
    参数：description - 用户要画的内容描述。"""
    chain = FLOWCHART_PROMPT | get_llm() | StrOutputParser()
    result = _safe_invoke(chain, description=description)
    return f"📊 流程图：\n\n{result}"


# =====================
# Tool 清单（注册给 LLM）
# =====================

# =====================
# Tool 9b: 抓取网页
# =====================

@tool
def fetch_url(url: str) -> str:
    """直接抓取指定网页 URL 的文本内容（不是搜索，是打开链接）。
    调用时机：用户给了具体的网址，想直接看网页内容时。
    参数：url - 要抓取的网页地址。"""
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urlparse, quote

    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"❌ 不支持的协议：{parsed.scheme}"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text()
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        return "\n".join(lines)[:5000]

    except Exception as e:
        return f"❌ 网页抓取失败：{e}"


ALL_TOOLS = [
    add_document,
    list_documents,
    remove_document,
    search_documents,
    summarize_document,
    compare_documents,
    write_related_work,
    web_search,
    fetch_url,
    generate_flowchart,
]

# 按名称查找 Tool
TOOL_BY_NAME = {tool.name: tool for tool in ALL_TOOLS}


def get_tool_by_name(name: str):
    """根据 tool 名称获取 tool 实例。"""
    return TOOL_BY_NAME.get(name)

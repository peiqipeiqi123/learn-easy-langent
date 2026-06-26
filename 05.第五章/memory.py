"""
第五章 学习助手 Agent — 记忆模块（分层摘要记忆）

设计：
  ┌─────────────────────────────────┐
  │         全局摘要                  │
  │  "用户研究主题、文档列表、对话脉络"  │
  ├─────────────────────────────────┤
  │  论文A摘要  │  论文B摘要  │  ...  │
  │  问题/方法   │  问题/方法   │      │
  │  /结论       │  /结论       │      │
  └─────────────────────────────────┘

用户可通过 show_memory / clear_memory 命令掌控记忆状态。
"""

import json
from datetime import datetime
from typing import Any
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# =====================
# 数据结构
# =====================

class DocumentStore:
    """管理已加载文档的注册信息和独立摘要。"""

    def __init__(self):
        # doc_id → {name, path, num_chunks, summary, loaded_at}
        self._docs: dict[str, dict[str, Any]] = {}

    # ---- 文档 CRUD ----

    def register(self, doc_id: str, name: str, path: str, num_chunks: int) -> None:
        """注册新加载的文档。"""
        self._docs[doc_id] = {
            "name": name,
            "path": path,
            "num_chunks": num_chunks,
            "summary": "",  # 文档独立摘要，后续由 LLM 填充
            "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    def remove(self, doc_id: str) -> dict | None:
        """移除文档，返回被删除的文档信息（供 FAISS 清理使用）。"""
        return self._docs.pop(doc_id, None)

    def update_summary(self, doc_id: str, summary: str) -> None:
        """更新某篇文档的独立摘要。"""
        if doc_id in self._docs:
            self._docs[doc_id]["summary"] = summary

    def list_all(self) -> list[dict]:
        """返回所有已加载文档的信息列表。"""
        return [
            {
                "id": doc_id,
                "name": info["name"],
                "num_chunks": info["num_chunks"],
                "summary": info["summary"] or "(暂无摘要)",
                "loaded_at": info["loaded_at"],
            }
            for doc_id, info in self._docs.items()
        ]

    def get_summaries(self) -> dict[str, str]:
        """获取所有文档摘要的字典 {doc_name: summary}。"""
        return {info["name"]: info["summary"] for info in self._docs.values()}

    def get(self, doc_id: str) -> dict | None:
        """获取指定文档的信息（安全访问）。"""
        return self._docs.get(doc_id)

    def items(self):
        """迭代所有文档 (doc_id, info) 对。"""
        return self._docs.items()

    def __len__(self) -> int:
        return len(self._docs)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._docs

    # ---- 序列化（方便调试 & 用户查看）----

    def to_dict(self) -> dict:
        return {
            doc_id: {
                "name": info["name"],
                "path": info["path"],
                "num_chunks": info["num_chunks"],
                "summary": info["summary"],
                "loaded_at": info["loaded_at"],
            }
            for doc_id, info in self._docs.items()
        }

    def to_display(self) -> str:
        """生成给用户看的文档列表文本。"""
        docs = self.list_all()
        if not docs:
            return "📭 当前没有已加载的文档。"
        lines = []
        for i, doc in enumerate(docs, 1):
            lines.append(f"\n{i}. **{doc['name']}**")
            lines.append(f"   - ID: `{doc['id']}`")
            lines.append(f"   - 分块数: {doc['num_chunks']}")
            lines.append(f"   - 摘要: {doc['summary']}")
            lines.append(f"   - 加载时间: {doc['loaded_at']}")
        return "\n".join(lines)


# =====================
# 分层记忆
# =====================

# 摘要生成提示模板
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是对话摘要助手。请根据以下对话历史，生成一段简洁的全局摘要（不超过300字）。

摘要需包含：
1. 用户当前的研究主题/目标
2. 已加载的文档列表
3. 最近讨论的核心问题和结论
4. 待跟进的事项（如有）

格式：一段连贯的中文段落，信息密度高，不要编号或列表。"""),
    ("human", "对话历史：\n{chat_history}\n\n请生成全局摘要："),
])


class LayeredMemory:
    """
    分层摘要记忆系统。

    - 全局摘要：对话的整体脉络（主题、文档列表、关键讨论）
    - 文档摘要：每篇文档的结构化摘要（独立槽位，不互相干扰）
    """

    def __init__(self):
        self.docs = DocumentStore()
        self.global_summary: str = ""
        self._chat_history: list[str] = []  # 原始对话记录（用于生成摘要）

    # ---- 对话记录 ----

    def add_turn(self, user_input: str, assistant_output: str) -> None:
        """记录一轮对话。"""
        self._chat_history.append(f"用户：{user_input}")
        self._chat_history.append(f"助手：{assistant_output}")

    def get_history_text(self, max_turns: int = 20) -> str:
        """获取最近 N 轮对话的文本（用于传入系统提示）。"""
        lines = self._chat_history[-(max_turns * 2):]  # 每轮 = 用户 + 助手 各1行
        return "\n".join(lines)

    # ---- 摘要更新 ----

    def auto_name(self, llm: ChatOpenAI) -> str | None:
        """根据最近对话自动生成会话名称（≤10字）。仅在会话未命名时调用。"""
        if not self._chat_history:
            return None
        prompt = ChatPromptTemplate.from_messages([
            ("system", "根据以下对话，生成一个简短的会话标题（不超过10个汉字）。只输出标题，不要引号、标点。"),
            ("human", "{history}\n\n标题："),
        ])
        chain = prompt | llm
        result = chain.invoke({"history": "\n".join(self._chat_history[-4:])})
        return result.content.strip()[:10] if hasattr(result, "content") else None

    def update_global_summary(self, llm: ChatOpenAI) -> str:
        """调用 LLM 重新生成全局摘要。"""
        if not self._chat_history:
            return ""
        chain = SUMMARY_PROMPT | llm
        result = chain.invoke({"chat_history": "\n".join(self._chat_history[-40:])})
        self.global_summary = result.content.strip()
        return self.global_summary

    def update_document_summary(
        self, doc_id: str, doc_content_snippet: str, llm: ChatOpenAI
    ) -> str:
        """为单篇文档生成结构化摘要（问题/方法/结论）。"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个学术论文分析助手。请根据提供的文档内容片段，生成结构化的文档摘要。

严格按以下 JSON 格式输出（不要输出其他内容）：
{{
    "title_guess": "根据内容推测的论文/文档标题",
    "problem": "文档解决的核心问题（一句话）",
    "method": "使用的核心方法/技术（2-3句话）",
    "key_findings": "关键发现或结论（2-3句话）",
    "keywords": ["关键词1", "关键词2", "关键词3"]
}}"""),
            ("human", "文档内容片段：\n{content}\n\n请生成结构化摘要："),
        ])
        chain = prompt | llm
        result = chain.invoke({"content": doc_content_snippet[:3000]})
        summary_text = result.content.strip()
        # 尝试解析 JSON，失败则保存原文
        try:
            parsed = json.loads(summary_text)
            formatted = (
                f"📄 {parsed.get('title_guess', doc_id)}\n"
                f"  - 问题: {parsed.get('problem', 'N/A')}\n"
                f"  - 方法: {parsed.get('method', 'N/A')}\n"
                f"  - 关键发现: {parsed.get('key_findings', 'N/A')}\n"
                f"  - 关键词: {', '.join(parsed.get('keywords', []))}"
            )
        except json.JSONDecodeError:
            formatted = summary_text
        self.docs.update_summary(doc_id, formatted)
        return formatted

    # ---- 展示 & 重置 ----

    def get_full_context(self) -> str:
        """
        组装完整的记忆上下文，供 System Prompt 使用。
        包含：全局摘要 + 各文档独立摘要。
        """
        parts = []
        if self.global_summary:
            parts.append(f"## 对话全局摘要\n{self.global_summary}")
        doc_summaries = self.docs.get_summaries()
        if doc_summaries:
            parts.append("\n## 已加载文档摘要")
            for name, summary in doc_summaries.items():
                parts.append(f"\n### {name}")
                parts.append(summary if summary else "(暂无摘要)")
        return "\n".join(parts) if parts else "(暂无记忆)"

    def to_display(self) -> str:
        """生成用户可读的记忆状态展示。"""
        lines = ["## 🧠 记忆状态\n"]
        lines.append(f"### 全局摘要\n{self.global_summary or '(暂无)'}")
        lines.append(f"\n### 文档列表 ({len(self.docs)} 篇)")
        lines.append(self.docs.to_display())
        lines.append(f"\n### 对话轮数\n{len(self._chat_history) // 2} 轮")
        return "\n".join(lines)

    def clear(self) -> None:
        """清空所有记忆和文档注册信息。"""
        self.global_summary = ""
        self._chat_history = []
        self.docs = DocumentStore()

    # ---- 持久化 ----

    def save_to_file(self, filepath: Path) -> None:
        """保存记忆状态到 JSON 文件（自动创建父目录）。"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "global_summary": self.global_summary,
            "chat_history": self._chat_history[-60:],  # 只保留最近 30 轮
            "docs": self.docs.to_dict(),
        }
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_from_file(self, filepath: Path) -> bool:
        """从 JSON 文件恢复记忆状态。返回是否成功。"""
        if not filepath.exists():
            return False
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            self.global_summary = data.get("global_summary", "")
            self._chat_history = data.get("chat_history", [])
            for doc_id, info in data.get("docs", {}).items():
                self.docs._docs[doc_id] = {  # 内部方法有权访问私有属性
                    "name": info["name"],
                    "path": info["path"],
                    "num_chunks": info["num_chunks"],
                    "summary": info.get("summary", ""),
                    "loaded_at": info.get("loaded_at", ""),
                }
            return True
        except Exception:
            return False


# =====================
# 全局实例
# =====================

_memory_instance: LayeredMemory | None = None


def get_memory() -> LayeredMemory:
    """获取全局记忆单例。"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = LayeredMemory()
    return _memory_instance


def reset_memory() -> LayeredMemory:
    """重置记忆（清空后返回新实例）。"""
    global _memory_instance
    if _memory_instance:
        _memory_instance.clear()
    _memory_instance = LayeredMemory()
    return _memory_instance

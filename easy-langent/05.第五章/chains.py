"""
第五章 学习助手 Agent — 链编排模块（备选方案 / 学习参考）

⚠️ 主程序 learning_assistant.py 采用消息列表 + stream() 方式直接调用 LLM，
   不使用此模块中的链。此文件保留供学习者对比两种实现方式：
   - chains.py: LCEL 链式编排（RunnablePassthrough + | 运算符）
   - learning_assistant.py: 消息列表 + 手动 Agent 循环（更灵活）

核心链结构（参考）：
  用户输入
    → 注入记忆上下文（RunnablePassthrough）
    → 主提示模板（SYSTEM_PROMPT + memory_context）
    → LLM + Tool 绑定（llm.bind_tools(ALL_TOOLS)）
    → 返回 AIMessage（可能包含 tool_calls）

如需在学习中使用此文件中的链，请从 learning_assistant 中导入 build_chain()，
但需自行实现 Tool 调用循环和流式输出。
"""

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import get_llm
from memory import get_memory  # 由 build_chain 中的 lambda 使用
from prompts import MAIN_PROMPT
from tools import ALL_TOOLS


def build_chain():
    """
    构建主链：

    结构:
      RunnablePassthrough.assign(memory_context=...)
      | MAIN_PROMPT
      | llm_with_tools
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    chain = (
        RunnablePassthrough.assign(
            memory_context=lambda x: get_memory().get_full_context()
        )
        | MAIN_PROMPT
        | llm_with_tools
    )

    return chain


def build_qa_chain():
    """
    构建无需 Tool 调用的纯问答链（用于 fallback 或简单对话）。

    结构:
      RunnablePassthrough.assign(memory_context=...)
      | MAIN_PROMPT（不含 tool binding）
      | llm
      | StrOutputParser
    """
    llm = get_llm()
    chain = (
        RunnablePassthrough.assign(
            memory_context=lambda x: get_memory().get_full_context()
        )
        | MAIN_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

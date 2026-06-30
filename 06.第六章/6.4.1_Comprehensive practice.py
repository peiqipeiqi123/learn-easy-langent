from typing import TypedDict
# 兼容低版本 Python 的 NotRequired 导入
try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver

# DeepSeek LLM（真实模型）
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 提前校验 API Key，避免运行时报错模糊
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("未在环境变量中检测到 API_KEY，请检查 .env 文件配置")

# 初始化真实 LLM
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/",
    model="deepseek-v4-flash",
    temperature=0.6
)

# ==========================================================
# 1. 定义 State（工作流共享状态 = Agent 内存）
# ==========================================================
class TextProcessState(TypedDict):
    """
    LangGraph 状态对象：
    用于在节点之间传递数据（类似全局共享内存）
    """
    raw_text: str                        # 输入：用户原始文本
    deduplicated_text: NotRequired[str]  # 过程：去重后的文本
    summary_text: NotRequired[str]       # 过程：LLM 生成摘要
    has_sensitive: NotRequired[bool]     # 过程：敏感词检测结果
    final_output: NotRequired[str]       # 输出：最终格式化结果


# ==========================================================
# 2. 定义节点函数（每个节点 = 一个处理模块）
# ==========================================================
def deduplicate_node(state: TextProcessState) -> TextProcessState:
    """文本去重节点：去除内容完全一致（忽略首尾空格）的重复行，丢弃纯空白行"""
    raw_text = state["raw_text"]
    lines = raw_text.split("\n")
    unique_lines = []
    seen = set()
    
    for line in lines:
        line_stripped = line.strip()
        # 过滤纯空白行，且未出现过该文本
        if line_stripped and line_stripped not in seen:
            seen.add(line_stripped)
            unique_lines.append(line)
    
    result_text = "\n".join(unique_lines)
    print("✅ 去重节点执行完成")
    
    # 复制原有状态，仅更新输出字段
    new_state = state.copy()
    new_state["deduplicated_text"] = result_text
    return new_state

def summary_node(state: TextProcessState) -> dict:
    """摘要生成节点（调用LLM）"""
    deduplicated_text = state["deduplicated_text"]
    prompt = PromptTemplate(
        input_variables=["text"],
        template="请为以下文本生成50字以内的简洁摘要，保留核心信息：\n{text}"
    )
    chain = prompt | llm
    summary = chain.invoke({"text": deduplicated_text}).content
    print("🤖 摘要节点执行完成")
    return {"summary_text": summary}

def sensitive_check_node(state: TextProcessState) -> dict:
    """敏感词检测节点"""
    summary = state["summary_text"]
    sensitive_words = ["敏感词1", "敏感词2", "违法", "违规"]
    has_sensitive = any(word in summary for word in sensitive_words)
    print("🔍 敏感词检测完成：", has_sensitive)
    return {"has_sensitive": has_sensitive}

def output_node(state: TextProcessState) -> dict:
    """输出节点（根据敏感词结果格式化）"""
    summary = state["summary_text"]
    has_sensitive = state["has_sensitive"]
    if has_sensitive:
        final_output = "⚠️ 检测到敏感内容，无法输出摘要"
    else:
        final_output = f"""✅ 文本处理完成
【摘要】
{summary}

【去重后原文】
{state['deduplicated_text']}
"""
    print("📤 输出节点执行完成")
    return {"final_output": final_output}


# ==========================================================
# 3. 构建线性工作流图（固定边）
# ==========================================================
def build_linear_graph():
    """构建线性 LangGraph 工作流，**实际启用状态历史**"""
    graph_builder = StateGraph(TextProcessState)

    # 注册节点
    graph_builder.add_node("deduplicate", deduplicate_node)
    graph_builder.add_node("summary", summary_node)
    graph_builder.add_node("sensitive_check", sensitive_check_node)
    graph_builder.add_node("output", output_node)

    # 配置固定边（线性执行）
    graph_builder.add_edge(START, "deduplicate")
    graph_builder.add_edge("deduplicate", "summary")
    graph_builder.add_edge("summary", "sensitive_check")
    graph_builder.add_edge("sensitive_check", "output")
    graph_builder.add_edge("output", END)

    # 编译时传入MemorySaver，真正启用状态历史
    # MemorySaver：内存级检查点，适合测试/开发，重启程序后状态丢失
    return graph_builder.compile(checkpointer=MemorySaver())


# ==========================================================
# 4. 测试运行（get_state_history 可正常使用）
# ==========================================================
if __name__ == "__main__":

    # 构建图
    linear_graph = build_linear_graph()

    # 初始状态（输入数据）
    test_state: TextProcessState = {
        "raw_text": "LangGraph是LangChain生态的工作流框架\nLangGraph支持状态管理\nLangGraph是LangChain生态的工作流框架\n支持动态分支和并行执行"
    }

    # thread_id：会话唯一标识，测试用随便命名，多会话用不同id即可
    config = {"configurable": {"thread_id": "text_process_test_001"}}

    # 执行工作流
    final_state = linear_graph.invoke(test_state, config=config)

    # 输出最终结果
    print("\n" + "=" * 50)
    print(final_state["final_output"])

    # 查看状态历史
    print("\n" + "=" * 50)
    history = list(linear_graph.get_state_history(config))
    print("状态快照数量：", len(history))
    print("提示：get_state_history 默认按时间倒序返回，最新状态排在最前\n")

    for i, snapshot in enumerate(history, 1):
        print(f"第{i}步快照：")
        print("状态数据：", snapshot.values)  
        print("下一节点：", snapshot.next)   
        print("-" * 30)
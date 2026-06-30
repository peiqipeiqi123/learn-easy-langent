
import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
# ------------------------------
# 1. 环境加载与模型初始化（保留你的DeepSeek配置，无任何修改）
# ------------------------------
load_dotenv()  # 加载.env中的API_KEY

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.3
)

# ------------------------------
# 2. 定义交互式状态（强类型，持久化存储所有流程数据）
# ------------------------------
class InteractiveOptState(TypedDict):
    user_input: str                # 固定：用户原始输入（全程不变）
    optimized_text: Optional[str]  # 动态：AI优化后文本（多轮更新）
    optimize_suggest: Optional[str]# 动态：优化建议/理由（多轮更新）
    user_feedback: Optional[str]   # 动态：用户反馈（确认/修改/退出）
    final_result: Optional[str]    # 最终：流程结束结果

# ------------------------------
# 3. 核心节点函数（无任何修改，保留你的原代码）
# ------------------------------
def optimize_node(state: InteractiveOptState) -> InteractiveOptState:
    """【机器节点】文本优化核心节点，使用管道符调用LLM"""
    user_input = state["user_input"]
    user_feedback = state["user_feedback"]

    if not user_feedback:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="请优化以下文本，提升流畅度和专业度，严格保留核心信息：\n{text}\n优化完成后，单独一行以【优化理由：】开头给出1-2条简洁优化原因"
        )
        chain = prompt | llm
        result = chain.invoke({"text": user_input}).content
    else:
        prompt = PromptTemplate(
            input_variables=["text", "feedback"],
            template="根据用户反馈针对性优化文本，严格保留核心信息：\n原文本：{text}\n用户反馈：{feedback}\n优化完成后，单独一行以【优化理由：】开头给出1-2条简洁优化原因"
        )
        chain = prompt | llm
        result = chain.invoke({"text": user_input, "feedback": user_feedback}).content

    split_flag = "【优化理由：】"
    if split_flag in result:
        optimized_text, optimize_suggest = result.split(split_flag, 1)
    else:
        optimized_text = result
        optimize_suggest = "AI未生成明确优化理由，建议重新优化"

    return {
        "optimized_text": optimized_text.strip(),
        "optimize_suggest": optimize_suggest.strip()
    }

def feedback_node(state: InteractiveOptState) -> InteractiveOptState:
    """【人机交互节点】展示结果+接收用户反馈，流程中断核心"""
    print("\n" + "-"*60)
    print("📝 AI优化后文本：")
    print(state["optimized_text"])
    print("\n💡 优化建议/理由：")
    print(state["optimize_suggest"])
    print("\n" + "-"*60)

    while True:
        user_feedback = input("请输入反馈（仅需输入：确认/修改/退出）：").strip()
        if user_feedback in ["确认", "修改", "退出"]:
            break
        print("❌ 输入无效！请严格输入「确认」「修改」「退出」，无其他字符\n")
    return {"user_feedback": user_feedback}

def feedback_router(state: InteractiveOptState) -> str:
    """【条件路由节点】循环核心，直接返回目标节点名（最新API要求）"""
    feedback = state["user_feedback"]
    if feedback == "确认":
        return "final"    # 确认→final节点
    elif feedback == "修改":
        return "optimize" # 修改→optimize节点（循环核心）
    else:
        return "exit"     # 退出→exit节点

def final_node(state: InteractiveOptState) -> InteractiveOptState:
    """【机器节点】流程正常结束，生成格式化结果"""
    final_result = (
        "✅ 【多轮文本优化流程完成】\n"
        f"📌 最终优化文本：\n{state['optimized_text']}\n"
        f"💡 优化核心总结：\n{state['optimize_suggest']}"
    )
    return {"final_result": final_result}

def exit_node(state: InteractiveOptState) -> InteractiveOptState:
    """【机器节点】用户主动退出，生成终止提示"""
    return {"final_result": "🔚 【文本优化流程终止】\n你主动退出，本次无最终优化结果"}

# ------------------------------
# 4. 搭建循环交互图（无任何修改，保留你的原代码）
# ------------------------------
def build_interactive_graph():
    """构建LangGraph循环状态图，彻底适配最新API终极规范"""
    graph_builder = StateGraph(InteractiveOptState)

    # 添加节点（无修改）
    graph_builder.add_node("optimize", optimize_node)
    graph_builder.add_node("feedback", feedback_node)
    graph_builder.add_node("final", final_node)
    graph_builder.add_node("exit", exit_node)

    # 配置普通边（无修改）
    graph_builder.add_edge(START, "optimize")
    graph_builder.add_edge("optimize", "feedback")

    # 适配最新API：source + path
    graph_builder.add_conditional_edges(
        source="feedback",  # 分支起始节点
        path=feedback_router  # 路由函数（直接返回目标节点名）
    )

    # 配置结束边（无修改）
    graph_builder.add_edge("final", END)
    graph_builder.add_edge("exit", END)

    # 编译图：开启状态持久化（多轮交互必需）
    return graph_builder.compile(checkpointer=MemorySaver())

# ------------------------------
# 5. 运行交互测试（★仅修改初始输入部分★，改为用户手动输入+非空校验）
# ------------------------------
if __name__ == "__main__":
    # 构建循环图（彻底解决所有API报错）
    interactive_graph = build_interactive_graph()
    print("🔧 多轮交互式文本优化工具已启动（适配LangGraph最新API）...\n")

    # ★核心修改：用户手动输入待优化句子 + 非空校验★
    print("="*40 + " 输入待优化句子 " + "="*40)
    while True:
        user_input_text = input("请输入需要AI优化的句子：").strip()
        if user_input_text:  # 非空校验，避免用户输入空内容
            break
        print("❌ 输入不能为空，请重新输入需要优化的句子！\n")

    # 初始状态：使用用户输入的句子，其余字段保持默认
    initial_state: InteractiveOptState = {
        "user_input": user_input_text,  # 替换为用户输入的内容
        "optimized_text": None,
        "optimize_suggest": None,
        "user_feedback": None,
        "final_result": None
    }

    # 启动多轮交互流程（保留你的config配置）
    print(f"\n🚀 已接收你的句子，开始第一轮AI优化...")
    config = {"configurable": {"thread_id": "text_process_test_001"}}
    final_state = interactive_graph.invoke(initial_state, config=config)

    # 展示最终结果
    print("\n" + "="*60)
    print(final_state["final_result"])
    print("="*60)

    # 展示交互轮次（状态持久化验证）
    history = list(interactive_graph.get_state_history(config))
    interact_rounds = len(history) // 2  # 每轮=优化节点+反馈节点
    print("状态快照数量（超步骤）：", len(history))

    # 保存可视化流程图（保留你的原代码）
    png_data = interactive_graph.get_graph().draw_mermaid_png()  # 获取PNG字节流
    with open("interactive_optimize_graph.png", "wb") as file:  # wb=二进制写入
        file.write(png_data)
    print("📊 工作流可视化图已保存：interactive_optimize_graph.png\n")
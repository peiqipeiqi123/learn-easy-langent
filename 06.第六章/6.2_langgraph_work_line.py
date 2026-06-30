from typing import TypedDict, NotRequired
from langgraph.graph import StateGraph

# ====== 定义状态（共享黑板）======
class TaskState(TypedDict):
    user_query: str #用户原始查询
    tool_result: NotRequired[str] #工具调用结果
    final_answer: NotRequired[str] #最终回答
    progress: NotRequired[int] #任务进度百分比

# ======节点函数=========
def parse_query(state: TaskState):
    print("\n====== 节点1 parse_query 输入状态 ======")
    print(state)

    query = state["user_query"]
    update = {
        "tool_result": f"已解析问题: {query}",
        "progress": 30
    }

    print("------ 节点1 更新字段 ------")
    print(update)

    return update


def call_tool(state: TaskState):
    print("\n====== 节点2 call_tool 输入状态 ======")
    print(state)

    result = f"工具搜索结果：关于『{state['user_query']}』的相关知识"
    update = {
        "tool_result": result,
        "progress": 70
    }

    print("------ 节点2 更新字段 ------")
    print(update)

    return update


def generate_answer(state: TaskState):
    print("\n====== 节点3 generate_answer 输入状态 ======")
    print(state)

    answer = f"最终回答：基于工具结果 -> {state['tool_result']}"
    update = {
        "final_answer": answer,
        "progress": 100
    }

    print("------ 节点3 更新字段 ------")
    print(update)

    return update

#========构建 LangGraph 工作流======

# 1. 创建图（绑定状态类型）
builder = StateGraph(TaskState)

# 2. 添加节点
builder.add_node("parse_query", parse_query)
builder.add_node("call_tool", call_tool)
builder.add_node("generate_answer", generate_answer)

# 3. 定义执行顺序（边）
builder.set_entry_point("parse_query")
builder.add_edge("parse_query", "call_tool")
builder.add_edge("call_tool", "generate_answer")

# 4. 编译图
graph = builder.compile()

# ====== 运行工作流 ======
init_state = TaskState(user_query="什么是 LangGraph？")

final_state = graph.invoke(init_state)

print("\n最终状态：")
print(final_state)
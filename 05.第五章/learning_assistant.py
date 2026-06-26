"""
第五章 学习助手 Agent — 主程序入口

用法：
    cd 05.第五章
    uv run python learning_assistant.py

特殊命令：
    :memory  或 :mem  → 查看当前记忆状态
    :reset   或 :clear → 重置所有记忆和文档
    :help             → 显示帮助
    q / quit / 退出    → 退出程序
"""

import json
import sys
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    HumanMessage,
    SystemMessage,
)

from config import get_llm, SESSIONS_DIR, DEFAULT_SESSION, PROJECT_ROOT
from memory import get_memory, reset_memory
from prompts import SYSTEM_PROMPT
from tools import (
    get_tool_by_name, ALL_TOOLS, save_chunks, load_chunks,
    set_session, get_session, rename_faiss,
)


# =====================
# 会话切换
# =====================

def _switch_session(name: str) -> str:
    """切换到指定会话。保存当前会话，加载目标会话。"""
    import tools as _t
    from config import OLD_SESSION_DIR

    # 1. 保存当前会话
    current = get_session()
    if current != name:
        memory = get_memory()
        session_file = SESSIONS_DIR / current / "session.json"
        try:
            memory.save_to_file(session_file)
            save_chunks()
        except Exception:
            pass

    # 2. 切换到目标会话
    target_dir = SESSIONS_DIR / name

    # 自动迁移旧版 .session/ → .sessions/default/
    if name == DEFAULT_SESSION and not (target_dir / "session.json").exists():
        old_file = OLD_SESSION_DIR / "session.json"
        if old_file.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(old_file, target_dir / "session.json")
            old_chunks = OLD_SESSION_DIR / "chunks.json"
            if old_chunks.exists():
                shutil.copy(old_chunks, target_dir / "chunks.json")
            # FAISS 由 _rebuild_faiss() 自动在新位置重建，无需手动迁移

    is_new = not (target_dir / "session.json").exists()

    _t.set_session(name)
    _t._all_chunks.clear()
    _t._vector_db = None
    reset_memory()

    if not is_new:
        memory = get_memory()
        ok = memory.load_from_file(target_dir / "session.json")
        chunks_ok = load_chunks(name)
        if ok and chunks_ok:
            _t._rebuild_faiss()
            return (
                f"✅ 已切换到「{name}」："
                f"{len(memory.docs)} 篇文档，"
                f"{len(memory._chat_history) // 2} 轮对话。"
            )
        else:
            return f"⚠️ 会话「{name}」数据不完整，已创建空会话。"

    return f"🆕 已创建新会话「{name}」。"


# =====================
# 命令处理
# =====================

def handle_command(user_input: str) -> str | None:
    """
    处理特殊命令。如果输入是命令，返回处理结果字符串；否则返回 None。

    ★ Insight ─────────────────────────────────────
    这类"元操作"（操作记忆而不是操作知识库）是 agent 项目
    的常见模式。它们不算 Tool（Tool 是给 LLM 调用的），
    而是用户在对话层直接触发的快捷指令。
    ─────────────────────────────────────────────────
    """
    cmd = user_input.strip().lower()

    if cmd in (":memory", ":mem", "查看记忆", "显示记忆"):
        return get_memory().to_display()

    # ---- 会话管理 ----

    if cmd == ":sessions" or cmd == "会话列表":
        import tools as _t
        sessions = sorted([
            d.name for d in SESSIONS_DIR.iterdir()
            if d.is_dir() and (d / "session.json").exists()
        ])
        if not sessions:
            return "📭 没有已保存的会话。"
        current = _t.get_session()
        lines = ["📂 已保存的会话："]
        for s in sessions:
            marker = " ← 当前" if s == current else ""
            # 读取会话信息
            try:
                data = json.loads((SESSIONS_DIR / s / "session.json").read_text("utf-8"))
                docs = len(data.get("docs", {}))
                turns = len(data.get("chat_history", [])) // 2
                lines.append(f"  • {s} （{docs}篇文档, {turns}轮）{marker}")
            except Exception:
                lines.append(f"  • {s}{marker}")
        return "\n".join(lines)

    if cmd.startswith(":session ") or cmd.startswith("会话 "):
        name = cmd.split(" ", 1)[1].strip()
        if not name:
            return "❌ 请输入会话名称。用法：:session <名称>"
        result = _switch_session(name)
        # 通知 main() 清空消息列表，防止旧会话消息泄漏
        return ("__CLEAR_MSGS__", result)

    if cmd in (":restore", ":restor", ":recover", "恢复会话"):
        return _switch_session(DEFAULT_SESSION)

    if cmd.startswith(":new ") or cmd.startswith("新会话 "):
        name = cmd.split(" ", 1)[1].strip()
        if not name:
            return "❌ 请输入会话名称。用法：:new <名称>"
        return _switch_session(name)

    if cmd.startswith(":rename ") or cmd.startswith("重命名 "):
        import shutil
        import tools as _t
        old = _t.get_session()
        name = cmd.split(" ", 1)[1].strip()
        if not name:
            return "❌ 请输入新名称。用法：:rename <新名称>"
        if (SESSIONS_DIR / name / "session.json").exists():
            return f"❌ 会话「{name}」已存在。"
        # 先保存，确保数据最新
        memory = get_memory()
        memory.save_to_file(SESSIONS_DIR / old / "session.json")
        save_chunks()
        # 关闭 FAISS 避免文件锁
        _t._vector_db = None
        # 复制而非移动（移动可能因文件锁失败）
        shutil.copytree(SESSIONS_DIR / old, SESSIONS_DIR / name)
        shutil.rmtree(SESSIONS_DIR / old)
        rename_faiss(old, name)
        _t.set_session(name)
        return f"✅ 已重命名「{old}」→「{name}」"

    if cmd in (":reset", ":clear", "清空记忆", "重置记忆"):
        reset_memory()
        import tools as _t
        _t._all_chunks.clear()
        _t._vector_db = None
        _t._rebuild_faiss()
        return "✅ 记忆和知识库已全部清空。"

    if cmd in (":help", ":h", "帮助", "help"):
        return (
            "## 📖 学习助手命令列表\n\n"
            "### 会话管理\n"
            "- `:sessions` / `会话列表` — 列出所有已保存的会话\n"
            "- `:session <名称>` / `会话 <名称>` — 进入指定会话（不存在则创建）\n"
            "- `:rename <新名称>` — 给当前会话改名\n"
            "- `:reset` / `:clear` — 清空当前会话\n\n"
            "### 其他\n"
            "- `:memory` / `:mem` — 查看当前记忆状态\n"
            "- `:help` — 显示此帮助\n"
            "- `q` / `quit` / `退出` — 退出程序\n\n"
            "### 使用提示\n"
            '- 先说「加载」文档的路径来添加文档\n'
            '- 直接提问即可在已加载文档中搜索\n'
            '- 试试说「比较文档A和文档B」\n'
            '- 试试说「帮我写 Related Work」\n'
            '- 试试说「画一个XX流程图」\n'
            '- 试试说「搜索一下最新的XX」'
        )

    if cmd in ("q", "quit", "退出", "exit"):
        return "__EXIT__"

    return None  # 不是命令，是普通对话


# =====================
# 消息助手
# =====================

def _build_system_message() -> SystemMessage:
    """
    用最新的记忆上下文构建系统消息。

    注意：使用 replace 而非 format，因为记忆内容中可能包含 JSON 的 {}
    字符，用 format 会导致 KeyError。
    """
    memory = get_memory()
    system_content = SYSTEM_PROMPT.replace(
        "{memory_context}", memory.get_full_context()
    )
    return SystemMessage(content=system_content)


def _execute_tool_calls(response: AIMessage) -> list[ToolMessage]:
    """
    执行 LLM 返回的所有 tool_calls，返回 ToolMessage 列表。

    ★ Insight ─────────────────────────────────────
    ToolMessage 的三个关键字段：
    - tool_call_id: 必须与 AIMessage 中的 tool_call id 匹配
    - name: 工具名称，帮助 LLM 理解"这个结果来自哪个工具"
    - content: 工具返回的结果（总是字符串）
    LLM 通过这些信息将"工具调用"和"工具结果"配对起来。
    ─────────────────────────────────────────────────
    """
    tool_messages: list[ToolMessage] = []

    for call in response.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        tool_id = call["id"]

        print(f"\n🔧 [调用工具: {tool_name}]")
        if tool_args:
            # 截断过长参数的显示
            args_display = str(tool_args)
            if len(args_display) > 100:
                args_display = args_display[:100] + "..."
            print(f"  参数: {args_display}")

        tool_func = get_tool_by_name(tool_name)
        if tool_func is None:
            result = f"❌ 未知工具：{tool_name}"
        else:
            try:
                result = tool_func.invoke(tool_args)
            except Exception as e:
                result = f"❌ 工具执行出错：{e}"

        # 截断过长结果的终端显示
        display = result[:250] + "..." if len(result) > 250 else result
        print(f"  结果: {display}")

        tool_messages.append(ToolMessage(
            tool_call_id=tool_id,
            name=tool_name,
            content=str(result),
        ))

    return tool_messages


# =====================
# 主循环
# =====================

def main():
    """主交互循环。

    ★ Insight ─────────────────────────────────────
    核心流程（基于消息列表的 Agent 模式）：
    1. 用户输入 → HumanMessage 加入消息列表
    2. LLM + bind_tools → 输出 AIMessage（可能有 tool_calls）
    3. 如有 tool_calls → 执行工具 → ToolMessage 加入列表 → LLM 再生成
    4. 如无 tool_calls → 直接输出

    消息列表让 LLM 能看到完整的"对话 → 工具调用 → 工具结果 → 回答"链，
    这是 LLM Agent 最基本也最重要的设计模式。
    ─────────────────────────────────────────────────
    """
    import tools as tools_module

    # 初始化 LLM（绑定 9 个 Tool）
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # ---- 启动默认会话，懒加载（秒开）----
    tools_module.set_session(DEFAULT_SESSION)
    tools_module._all_chunks.clear()
    tools_module._vector_db = None
    reset_memory()

    # ---- 显示可用的历史会话（就像侧边栏）----
    saved = sorted([
        d.name for d in SESSIONS_DIR.iterdir()
        if d.is_dir() and (d / "session.json").exists()
    ])
    if saved:
        print("\n📂 之前的对话：")
        for s in saved:
            try:
                data = json.loads((SESSIONS_DIR / s / "session.json").read_text("utf-8"))
                docs = len(data.get("docs", {}))
                turns = len(data.get("chat_history", [])) // 2
                print(f"  · {s}（{docs}篇文档，{turns}轮）")
            except Exception:
                print(f"  · {s}")
        print("  输入 :session <名称> 进入 → 就像点开之前的聊天窗口")

    # 对话消息历史（不含 SystemMessage，因为 SystemMessage 每轮重建以注入最新记忆）
    MAX_MESSAGES = 60  # 保留最近约 60 条消息，防止长会话爆 Token
    messages: list = []

    # 打印欢迎信息
    print("=" * 60)
    print("  🎓 学习助手 Agent (Learning Assistant)")
    print("  基于 LangChain + DeepSeek + FAISS + Qwen3 Embedding")
    print("=" * 60)
    print("  📄 随时告诉我文档路径，我来帮你加载和分析")
    print("  ❓ 直接提问，我会在已加载的文档中搜索答案")
    print("  🔧 输入 :help 查看命令列表")
    print("  🚪 输入 q / quit / 退出 结束对话")
    print("=" * 60)

    turn = 0
    while True:
        # ---- 用户输入 ----
        try:
            user_input = input(f"\n👤 [{get_session()}:{turn}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        # ---- 检查特殊命令 ----
        cmd_result = handle_command(user_input)
        if cmd_result == "__EXIT__":
            print("👋 再见！")
            break
        if cmd_result is not None:
            if isinstance(cmd_result, tuple) and cmd_result[0] == "__CLEAR_MSGS__":
                messages.clear()
                print(f"\n{cmd_result[1]}")
            else:
                print(f"\n{cmd_result}")
            continue

        try:
            turn += 1

            messages.append(HumanMessage(content=user_input))

            # ---- Agent 循环 ----
            MAX_TOOL_ITERATIONS = 8
            output = None
            started_output = False

            for iteration in range(MAX_TOOL_ITERATIONS):
                system_msg = _build_system_message()
                full_messages = [system_msg] + messages

                chunks: list = []
                has_tool_calls = False

                for chunk in llm_with_tools.stream(full_messages):
                    chunks.append(chunk)
                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        has_tool_calls = True
                    if chunk.content and not has_tool_calls:
                        if not started_output:
                            print("\n🤖 助手：", end=" ", flush=True)
                            started_output = True
                        print(chunk.content, end="", flush=True)

                if chunks:
                    merged = chunks[0]
                    for c in chunks[1:]:
                        merged = merged + c
                    response = merged
                else:
                    output = "（Agent 未生成回复，请重试）"
                    break

                if has_tool_calls and hasattr(response, "tool_calls") and response.tool_calls:
                    if started_output:
                        print()
                    messages.append(response)
                    tool_messages = _execute_tool_calls(response)
                    messages.extend(tool_messages)
                else:
                    if started_output:
                        print()
                    messages.append(response)
                    output = response.content if hasattr(response, "content") else str(response)
                    break
            else:
                output = "⚠️ Tool 调用次数过多，已中止。"

            if output is None:
                output = "（Agent 未生成回复，请重试）"

            if not started_output and output:
                print(f"\n🤖 助手：\n{output}")

        except Exception as e:
            output = f"⚠️ 程序内部错误：{e}\n   请将此错误信息发送给开发者。"
            print(f"\n🤖 助手：\n{output}")
            import traceback
            traceback.print_exc()

        # ---- 更新分层记忆 ----
        memory = get_memory()
        memory.add_turn(user_input, output)

        # 首次有实质对话后，自动给默认会话取名
        if get_session() == DEFAULT_SESSION and turn == 1:
            try:
                new_name = memory.auto_name(get_llm())
                if new_name and not (SESSIONS_DIR / new_name / "session.json").exists():
                    import shutil
                    memory.save_to_file(SESSIONS_DIR / DEFAULT_SESSION / "session.json")
                    save_chunks()
                    shutil.copytree(SESSIONS_DIR / DEFAULT_SESSION, SESSIONS_DIR / new_name)
                    shutil.rmtree(SESSIONS_DIR / DEFAULT_SESSION)
                    rename_faiss(DEFAULT_SESSION, new_name)
                    tools_module.set_session(new_name)
                    print(f"\n✨ 会话自动命名为「{new_name}」")
            except Exception:
                pass

        # 每 3 轮自动更新一次全局摘要
        if turn % 3 == 0:
            try:
                memory.update_global_summary(get_llm())
            except Exception as e:
                print(f"  [后台] 摘要更新失败：{e}", flush=True)

        # ---- 自动持久化（保存到当前会话目录）----
        try:
            session_file = SESSIONS_DIR / get_session() / "session.json"
            memory.save_to_file(session_file)
            save_chunks()
        except Exception as e:
            print(f"  [后台] 保存失败：{e}", flush=True)

        # ---- 消息列表截断（按 HumanMessage 边界，不破坏 Tool 调用对）----
        if len(messages) > MAX_MESSAGES:
            # 找到倒数第 MAX_MESSAGES 个 HumanMessage，从那里开始截
            human_indices = [
                i for i, m in enumerate(messages)
                if isinstance(m, HumanMessage)
            ]
            cutoff = max(0, len(human_indices) - (MAX_MESSAGES // 3))
            if cutoff < len(human_indices):
                messages = messages[human_indices[cutoff]:]


# =====================
# 入口
# =====================

if __name__ == "__main__":
    main()

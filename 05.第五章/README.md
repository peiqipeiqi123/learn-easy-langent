# 第五章 课程中期综合实践：学习助手 Agent

> **一句话定义**：一个能动态加载论文/文档、RAG 问答、对比分析、写 Related Work、联网搜索、抓取网页和生成流程图的智能学习助手。

## 项目背景与使用场景

**目标用户**：研究者 / 学习者

**核心场景**：
- 📄 拖入 PDF → 立即提问 → 带来源引用的回答
- 🔬 加载两篇论文 → 对比异同 → Mermaid 对比图
- ✍️ 基于已加载文献写 Related Work 段落
- 🌐 联网搜索最新信息 + 直接抓取指定网页
- 📊 生成 Mermaid 流程图 / 关系图

## 系统架构

```
用户输入 → [命令处理] → 消息列表 + 分层记忆
                       → LLM + 10 Tool 绑定
                       → Tool 执行循环（最多8轮）
                       → 流式输出 → 自动保存
```

## 功能清单

### 10 个 Tool

| Tool | 功能 | 触发 |
|------|------|------|
| `add_document` | 加载 PDF/TXT/DOCX/MD（支持文件夹批量） | "加载这篇论文" |
| `list_documents` | 列出已加载文档及摘要 | "看看有哪些文档" |
| `remove_document` | 移除指定文档 | "删掉那篇" |
| `search_documents` | RAG 检索 + LLM 生成带来源引用的回答 | 基于文档的任意提问 |
| `summarize_document` | 结构化摘要（JSON：问题/方法/结论/关键词） | "总结这篇" |
| `compare_documents` | 两篇文档对比 + Mermaid 关系图 | "比较A和B" |
| `write_related_work` | 基于已加载文献撰写 Related Work 段落 | "帮我写 Related Work" |
| `web_search` | DuckDuckGo 联网搜索 | "搜索一下最新的XX" |
| `fetch_url` | 抓取指定网页内容 | "帮我看看这个链接" |
| `generate_flowchart` | 生成 Mermaid 流程图 | "画个结构图" |

### 分层记忆

```
全局摘要（对话脉络 + 文档列表 + 关键讨论）
  ├── 文档A 摘要（问题/方法/结论/关键词）
  ├── 文档B 摘要
  └── ...
```

### 多会话管理

| 命令 | 作用 |
|------|------|
| `:sessions` | 列出所有已保存的会话 |
| `:session <名称>` | 进入指定会话（不存在则创建） |
| `:rename <名称>` | 重命名当前会话 |
| `:reset` | 清空当前会话 |
| `:memory` | 查看当前记忆状态 |
| `:help` | 命令列表 |

- 启动后自动显示历史会话列表
- 首次有实质对话后自动生成会话名（如 "研究问题选择之道"）
- 每次退出自动保存，下次可直接进入之前的对话

### 流式输出

回答逐字输出，不再等待完整生成。

## 项目结构

```
05.第五章/
  ├── README.md              ← 你在这里
  ├── config.py              ← 环境变量、LLM/Embedding 初始化
  ├── memory.py              ← 分层摘要记忆 + 自动命名
  ├── prompts.py             ← 系统提示 + 5 个任务专用模板
  ├── tools.py               ← 10 个 Tool 定义
  ├── chains.py              ← LCEL 链式编排（备选方案/学习参考）
  └── learning_assistant.py  ← 主程序入口
```

## 环境配置与运行

### 前置条件

1. Python 3.13+，`uv` 管理依赖
2. 项目根目录 `.env` 已配置 DeepSeek API Key
3. Qwen3 嵌入模型已下载（运行过第四章 4.4.3）
4. `uv sync` 安装依赖

### 运行

```bash
cd 05.第五章
uv run python learning_assistant.py
```

## 与课程要求对照

| 课程要求 | 本项目实现 |
|---------|-----------|
| 使用 PromptTemplate（参数化/少样本） | ✅ 系统提示 + 5 个任务专用模板 |
| Runnable / \| 构建 ≥2 步链 | ✅ LCEL 链 + 消息列表 + 多步 Tool 调用 |
| Memory 或 Tool 二选一 | ✅ 分层摘要 Memory + 10 个 Tool |
| OutputParser 或结构化输出 | ✅ 分析任务 JSON，问答 Markdown，流程图 Mermaid |

## 已知限制

- **图片不支持**：`PDFPlumberLoader` 只提取文字，图表会被跳过。如需图片分析可换多模态模型（GPT-5.5、Claude Opus 4.8、豆包 2.1 Pro、Qwen3.7 Max）
- **仅支持 PDF/TXT/MD/DOCX**：不支持 .doc（旧版 Word 二进制格式）、.epub 等
- **FAISS 仅 CPU**：无 GPU 加速

## 免责声明

本项目仅供个人学习 LangChain 框架使用。

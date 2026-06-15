# ============================================================
# 免责声明：本代码仅供个人学习 LangChain 工具调用使用。
# 严禁用于：非法爬取、侵犯隐私、恶意攻击等违法行为。
# 使用者需遵守目标网站的 robots.txt 及当地法律法规。
# ============================================================
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import ipaddress
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

  # 加载密钥
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

# ======================
# 抓取网页工具
# ======================

# 安全校验：拦截内网地址，防止 SSRF 攻击
BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "[::1]"}

def _validate_url(url: str) -> str | None:
    """校验 URL 安全性，返回错误信息或 None（通过）"""
    parsed = urlparse(url)
    # 1. 只允许 http/https
    if parsed.scheme not in ("http", "https"):
        return f"不支持的协议：{parsed.scheme}"
    # 2. 拦截裸 IP
    hostname = parsed.hostname
    if hostname is None:
        return "无法解析 URL 主机名"
    # 3. 拦截内网地址
    if hostname.lower() in BLOCKED_HOSTS:
        return f"禁止访问内网地址：{hostname}"
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback:
            return f"禁止访问内网地址：{hostname}"
    except ValueError:
        pass  # 不是 IP 地址，是域名，放行
    return None  # 通过


@tool
def fetch_webpage(url: str) -> str:
    """抓取指定网页的文本内容。仅支持 http/https 公网地址，禁止内网访问。"""
    # 安全校验
    error = _validate_url(url)
    if error:
        return f"安全拦截：{error}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()  # 非 200 状态码会抛异常
        resp.encoding = resp.apparent_encoding  # 自动检测编码

        soup = BeautifulSoup(resp.text, "html.parser")
        # 去掉 script 和 style 标签
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text()
        # 合并多余空行
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)[:5000]  # 限制长度，避免超出 Token

    except requests.RequestException as e:
        return f"抓取失败：{e}"


tools = [fetch_webpage]

# 绑定工具到 LLM（让 LLM 知道有 fetch_webpage 可用）
llm_with_tools = llm.bind_tools(tools)

# ======================
# 摘要记忆
# ======================
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是对话摘要助手，需简洁总结以下对话的核心信息（包含网页主题、用户问题、关键结论），不超过80字。"),
    ("human", "对话历史：{chat_history_text}\n请生成摘要：")
])

summary_chain = summary_prompt | llm

memory_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是友好的网页内容查询助手，规则如下：
1. 当用户输入一个 URL 时，调用 fetch_webpage 工具抓取网页内容。
2. 仅抓取用户明确提供的 URL，不要自行搜索其他网站。
3. 当用户针对网页内容提问时，基于抓取到的内容回答问题。
4. 回答简洁清晰，用中文。"""),
    ("system", "对话摘要：{chat_summary}"),
    ("human", "{user_input}")
])

base_chain = (
    RunnablePassthrough.assign(
        chat_summary=lambda x: summary_chain.invoke(
            {"chat_history_text": "\n".join(
                [f"{msg.type}: {msg.content}" for msg in x["chat_history"]]
            )}
        ).content
    )
    | memory_prompt
    | llm_with_tools  # 使用绑定工具的 LLM
)

memory_store = {}

def get_memory_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = InMemoryChatMessageHistory()
    return memory_store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_memory_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"
)


# ======================
# 测试
# ======================
if __name__ == "__main__":
    from langchain_core.messages import AIMessage, ToolMessage

    config = {"configurable": {"session_id": "web_user_001"}}

    print("===== 网页内容查询助手 =====")
    print("提示：输入 URL 可抓取网页内容，然后针对内容提问。输入 q 退出\n")

    while True:
        user_input = input("请输入您想提问的信息：")
        if user_input.lower() in ("q", "quit", "退出"):
            print("助手：再见！")
            break

        response = chain_with_memory.invoke(
            {"user_input": user_input},
            config=config
        )

        # 检查模型是否决定调用工具
        if isinstance(response, AIMessage) and response.tool_calls:
            history = get_memory_history("web_user_001")
            for call in response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                print(f"[调用工具: {tool_name}]")

                tool_func = next(t for t in tools if t.name == tool_name)
                result = tool_func.invoke(tool_args)

                print(f"[工具返回: {result[:200]}...]" if len(result) > 200 else f"[工具返回: {result}]")

                history.add_message(ToolMessage(
                    tool_call_id=call["id"],
                    name=tool_name,
                    content=str(result)
                ))

            # 工具执行完后，再次调用 LLM 生成最终回答
            final_response = chain_with_memory.invoke(
                {"user_input": "已抓取网页内容，请总结并回答用户刚才的问题。"},
                config=config
            )
            print(f"助手：{final_response.content}\n")
        else:
            print(f"助手：{response.content}\n")
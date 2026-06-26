"""
第五章 学习助手 Agent — 配置模块
负责：环境变量加载、LLM 初始化、嵌入模型初始化、路径配置

运行前确保 .env 文件中有 API_KEY 和 BASE_URL
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# =====================
# 环境变量 & 路径
# =====================

# 项目根目录（05.第五章 的父目录）
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 从项目根目录加载 .env
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 第五章目录（当前文件所在目录）
CHAPTER5_DIR = Path(__file__).parent

# 复用第四章的 Qwen 嵌入模型路径
# 第四章模型位置：04.第四章/4.3_rag/models/Qwen
MODEL_PATH = CHAPTER5_DIR.parent / "04.第四章" / "4.3_rag" / "models" / "Qwen"

# 第五章知识库目录（用户上传的文档存这里）
KNOWLEDGE_BASE_DIR = CHAPTER5_DIR / "knowledge_base"
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)

# 多会话持久化目录（每个会话独立存储）
SESSIONS_DIR = CHAPTER5_DIR / ".sessions"
SESSIONS_DIR.mkdir(exist_ok=True)
DEFAULT_SESSION = "default"

# 兼容旧版单会话目录（迁移用）
OLD_SESSION_DIR = CHAPTER5_DIR / ".session"


# =====================
# LLM 初始化
# =====================

def init_llm(temperature: float = 0.3) -> ChatOpenAI:
    """初始化 DeepSeek Chat 模型（兼容 OpenAI 接口）。"""
    if not API_KEY:
        raise ValueError("未找到 API_KEY，请检查项目根目录 .env 文件")
    return ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model="deepseek-chat",
        temperature=temperature,
        timeout=30,
        max_retries=2,
    )


# =====================
# 嵌入模型初始化
# =====================

def init_embeddings() -> HuggingFaceEmbeddings:
    """初始化本地 Qwen3 嵌入模型（CPU 运行）。"""
    model_path_str = str(MODEL_PATH.resolve())
    if not os.path.exists(model_path_str):
        raise FileNotFoundError(
            f"嵌入模型路径不存在：{model_path_str}\n"
            f"请先运行第四章 4.4.3 下载 Qwen3-Embedding 模型"
        )
    return HuggingFaceEmbeddings(
        model_name=model_path_str,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# =====================
# 懒加载单例（避免重复初始化）
# =====================

_llm_instance: ChatOpenAI | None = None
_embeddings_instance: HuggingFaceEmbeddings | None = None


def get_llm() -> ChatOpenAI:
    """获取 LLM 单例。"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = init_llm()
    return _llm_instance


def get_embeddings() -> HuggingFaceEmbeddings:
    """获取嵌入模型单例。"""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = init_embeddings()
    return _embeddings_instance

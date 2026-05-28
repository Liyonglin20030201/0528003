"""项目配置文件"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# 文档目录
DOCS_DIR = BASE_DIR / "docs"

# 向量数据库存储目录
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# 文本切分参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# 检索参数
RETRIEVAL_TOP_K = 4

# Embedding 模型（使用 HuggingFace 中文 embedding 模型）
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"

# Qwen 模型配置（通过 Ollama 调用）
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL_NAME = "qwen2.5:1.5b"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024

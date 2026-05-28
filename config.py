"""项目配置文件

从 config.yaml 加载配置，对外暴露与原来相同的常量接口。
"""
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.yaml"


def _load_yaml() -> dict:
    """加载 YAML 配置文件"""
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_cfg = _load_yaml()

# 目录
DOCS_DIR = BASE_DIR / "docs"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# 文本切分参数
CHUNK_SIZE: int = _cfg["chunk_size"]
CHUNK_OVERLAP: int = _cfg["chunk_overlap"]

# 检索参数
RETRIEVAL_TOP_K: int = _cfg["retrieval_top_k"]

# Embedding 模型
EMBEDDING_MODEL_NAME: str = _cfg["embedding_model_name"]

# LLM 配置
LLM_BASE_URL: str = _cfg["llm"]["base_url"]
LLM_MODEL_NAME: str = _cfg["llm"]["model_name"]
LLM_TEMPERATURE: float = _cfg["llm"]["temperature"]
LLM_MAX_TOKENS: int = _cfg["llm"]["max_tokens"]

# Ollama 服务配置
OLLAMA_HEALTH_URL: str = _cfg["ollama"]["health_url"]
OLLAMA_MAX_RETRIES: int = _cfg["ollama"]["max_retries"]
OLLAMA_RETRY_DELAY: int = _cfg["ollama"]["retry_delay"]

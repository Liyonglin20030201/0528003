"""模块四：大模型推理模块

使用 Qwen 量化模型通过 Ollama 生成回答。
"""
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage

import config


def get_llm() -> ChatOpenAI:
    """获取 Qwen LLM 实例（通过 Ollama 的 OpenAI 兼容接口）"""
    llm = ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key="ollama",
        model=config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    return llm


def generate_response(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
) -> str:
    """调用大模型生成回答"""
    response = llm.invoke(messages)
    return response.content

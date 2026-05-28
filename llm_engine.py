"""模块四：大模型推理模块

使用 Qwen 量化模型通过 Ollama 生成回答。
包含 Ollama 服务健康检查与重试机制。
"""
import time

import httpx
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage

import config


class OllamaConnectionError(Exception):
    """Ollama 服务不可用异常"""
    pass


def check_ollama_health() -> bool:
    """检查 Ollama 服务是否可用"""
    try:
        resp = httpx.get(config.OLLAMA_HEALTH_URL, timeout=5)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def wait_for_ollama() -> None:
    """等待 Ollama 服务就绪，使用指数退避重试

    Raises:
        OllamaConnectionError: 重试耗尽后仍无法连接
    """
    for attempt in range(1, config.OLLAMA_MAX_RETRIES + 1):
        if check_ollama_health():
            return
        if attempt < config.OLLAMA_MAX_RETRIES:
            delay = config.OLLAMA_RETRY_DELAY * (2 ** (attempt - 1))
            print(
                f"[LLM] Ollama 服务暂不可用，第 {attempt} 次重试，"
                f"等待 {delay} 秒..."
            )
            time.sleep(delay)

    raise OllamaConnectionError(
        f"无法连接 Ollama 服务（已重试 {config.OLLAMA_MAX_RETRIES} 次）。\n"
        f"请确认：\n"
        f"  1. Ollama 已安装并启动（ollama serve）\n"
        f"  2. 模型已拉取（ollama pull {config.LLM_MODEL_NAME}）\n"
        f"  3. 服务地址正确（当前: {config.OLLAMA_HEALTH_URL}）"
    )


def get_llm() -> ChatOpenAI:
    """获取 Qwen LLM 实例（通过 Ollama 的 OpenAI 兼容接口）

    启动前自动检测 Ollama 服务可用性。
    """
    wait_for_ollama()
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
    """调用大模型生成回答，带连接异常友好提示"""
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        if not check_ollama_health():
            raise OllamaConnectionError(
                "Ollama 服务在推理过程中断开连接，请检查服务状态后重试。"
            ) from e
        raise

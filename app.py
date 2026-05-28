"""模块五：Streamlit 聊天界面

支持多轮连续对话的校园问答助手前端。
"""
import streamlit as st
from langchain.schema import HumanMessage, AIMessage

from knowledge_base import get_embeddings, load_vector_store, init_knowledge_base
from retriever import Retriever
from prompt_builder import build_prompt, format_context
from llm_engine import get_llm
import config


def initialize_system():
    """初始化系统组件（仅执行一次）"""
    if "initialized" not in st.session_state:
        with st.spinner("正在加载系统组件..."):
            if not config.VECTOR_STORE_DIR.exists():
                st.info("首次运行，正在构建知识库...")
                init_knowledge_base()

            embeddings = get_embeddings()
            vector_store = load_vector_store(embeddings)
            st.session_state.retriever = Retriever(vector_store)
            st.session_state.llm = get_llm()
            st.session_state.prompt_template = build_prompt()
            st.session_state.initialized = True


def get_chat_history() -> list:
    """将 Streamlit 消息历史转为 LangChain 消息格式"""
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


def process_query(query: str) -> str:
    """处理用户问题：检索 → 组装提示 → 生成回答"""
    docs = st.session_state.retriever.retrieve(query)
    context = format_context(docs)
    chat_history = get_chat_history()

    messages = st.session_state.prompt_template.format_messages(
        context=context,
        chat_history=chat_history,
        question=query,
    )

    response = st.session_state.llm.invoke(messages)
    return response.content


def main():
    st.set_page_config(
        page_title="华智大学智能问答助手",
        page_icon="🎓",
        layout="centered",
    )

    st.title("🎓 华智大学智能问答助手")
    st.caption("我可以帮你解答关于奖学金、选课、研究生培养等校园规章制度方面的问题")

    initialize_system()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("请输入你的问题，例如：博士生国家奖学金多少钱？"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = process_query(query)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar:
        st.header("系统信息")
        st.markdown(f"**模型:** `{config.LLM_MODEL_NAME}`")
        st.markdown(f"**Embedding:** `{config.EMBEDDING_MODEL_NAME}`")
        st.markdown(f"**检索 Top-K:** `{config.RETRIEVAL_TOP_K}`")

        if st.button("清空对话历史"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.markdown("### 示例问题")
        examples = [
            "博士研究生国家奖学金奖励多少钱？",
            "选课每学期最多能选多少学分？",
            "研究生开题报告什么时候完成？",
            "国家助学金每月发多少？",
            "重修课程怎么选？",
        ]
        for ex in examples:
            if st.button(ex, key=ex):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()


if __name__ == "__main__":
    main()

"""模块五：Streamlit 聊天界面

支持多轮连续对话的校园问答助手前端。
集成 Ollama 服务状态检测与知识库管理功能。
"""
import streamlit as st
from langchain.schema import HumanMessage, AIMessage

from knowledge_base import (
    get_embeddings,
    load_vector_store,
    init_knowledge_base,
    rebuild_knowledge_base,
    list_documents,
    add_document,
    delete_document,
)
from retriever import Retriever
from prompt_builder import build_prompt, format_context
from llm_engine import get_llm, check_ollama_health, OllamaConnectionError
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
            st.session_state.prompt_template = build_prompt()

            try:
                st.session_state.llm = get_llm()
                st.session_state.ollama_ok = True
            except OllamaConnectionError as e:
                st.session_state.llm = None
                st.session_state.ollama_ok = False
                st.session_state.ollama_error = str(e)

            st.session_state.initialized = True


def try_reconnect_ollama():
    """尝试重新连接 Ollama 服务"""
    try:
        st.session_state.llm = get_llm()
        st.session_state.ollama_ok = True
        st.session_state.ollama_error = None
    except OllamaConnectionError as e:
        st.session_state.ollama_ok = False
        st.session_state.ollama_error = str(e)


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
    if not st.session_state.ollama_ok:
        try_reconnect_ollama()
        if not st.session_state.ollama_ok:
            return (
                "⚠️ 无法连接到 Ollama 服务，请确认服务已启动后重试。\n\n"
                f"错误详情：{st.session_state.ollama_error}"
            )

    docs = st.session_state.retriever.retrieve(query)
    context = format_context(docs)
    chat_history = get_chat_history()

    messages = st.session_state.prompt_template.format_messages(
        context=context,
        chat_history=chat_history,
        question=query,
    )

    try:
        response = st.session_state.llm.invoke(messages)
        return response.content
    except Exception:
        if not check_ollama_health():
            st.session_state.ollama_ok = False
            return (
                "⚠️ Ollama 服务在推理过程中断开连接。\n\n"
                "请检查服务状态后点击侧边栏「重新连接」按钮。"
            )
        raise


def render_kb_management():
    """渲染知识库管理界面"""
    st.header("知识库管理")

    docs = list_documents()
    st.markdown(f"当前文档数量：**{len(docs)}**")

    # 文档列表与删除
    if docs:
        with st.expander("查看/删除现有文档", expanded=False):
            for doc_name in docs:
                col1, col2 = st.columns([3, 1])
                col1.text(doc_name)
                if col2.button("删除", key=f"del_{doc_name}"):
                    delete_document(doc_name)
                    st.success(f"已删除: {doc_name}")
                    st.rerun()

    # 上传新文档
    st.subheader("添加文档")
    uploaded_file = st.file_uploader(
        "上传 .txt 文件",
        type=["txt"],
        key="doc_uploader",
    )
    if uploaded_file is not None:
        if st.button("确认添加"):
            content = uploaded_file.read().decode("utf-8")
            add_document(uploaded_file.name, content)
            st.success(f"已添加: {uploaded_file.name}")
            st.rerun()

    # 重新构建知识库
    st.subheader("重建知识库")
    st.caption("添加或删除文档后，需重建知识库使更改生效")
    if st.button("重新构建知识库"):
        with st.spinner("正在重新构建知识库..."):
            vector_store = rebuild_knowledge_base()
            st.session_state.retriever = Retriever(vector_store)
        st.success("知识库重建完成！")


def main():
    st.set_page_config(
        page_title="华智大学智能问答助手",
        page_icon="🎓",
        layout="centered",
    )

    st.title("🎓 华智大学智能问答助手")
    st.caption("我可以帮你解答关于奖学金、选课、研究生培养等校园规章制度方面的问题")

    initialize_system()

    # Ollama 服务状态提示
    if not st.session_state.ollama_ok:
        st.warning(
            "⚠️ Ollama 服务未就绪，问答功能暂不可用。\n\n"
            "请启动 Ollama 后点击侧边栏「重新连接」按钮。",
            icon="🔌",
        )

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
        # 系统信息
        st.header("系统信息")
        st.markdown(f"**模型:** `{config.LLM_MODEL_NAME}`")
        st.markdown(f"**Embedding:** `{config.EMBEDDING_MODEL_NAME}`")
        st.markdown(f"**检索 Top-K:** `{config.RETRIEVAL_TOP_K}`")

        # Ollama 连接状态
        status = "🟢 已连接" if st.session_state.ollama_ok else "🔴 未连接"
        st.markdown(f"**Ollama 状态:** {status}")
        if not st.session_state.ollama_ok:
            if st.button("重新连接"):
                with st.spinner("正在连接 Ollama..."):
                    try_reconnect_ollama()
                if st.session_state.ollama_ok:
                    st.success("连接成功！")
                else:
                    st.error("连接失败，请确认 Ollama 已启动。")
                st.rerun()

        if st.button("清空对话历史"):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # 知识库管理
        render_kb_management()

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

"""模块五：Streamlit 聊天界面

支持多轮连续对话的校园问答助手前端。
集成 Ollama 服务状态检测、知识库管理、对话历史保存、引用来源显示。
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
from chat_history import (
    list_sessions,
    load_session,
    save_session,
    delete_session,
    rename_session,
    export_session_as_text,
    new_session_id,
)
from doc_parser import extract_text, SUPPORTED_EXTENSIONS
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

    if "session_id" not in st.session_state:
        st.session_state.session_id = new_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = {}


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


def process_query(query: str) -> tuple[str, list[dict]]:
    """处理用户问题：检索 → 组装提示 → 生成回答

    Returns:
        (回答文本, 引用来源列表)
    """
    if not st.session_state.ollama_ok:
        try_reconnect_ollama()
        if not st.session_state.ollama_ok:
            return (
                "⚠️ 无法连接到 Ollama 服务，请确认服务已启动后重试。\n\n"
                f"错误详情：{st.session_state.ollama_error}"
            ), []

    docs = st.session_state.retriever.retrieve(query)
    context = format_context(docs)
    chat_history = get_chat_history()

    # 构建引用来源信息
    sources = []
    for i, doc in enumerate(docs, 1):
        source_path = doc.metadata.get("source", "未知来源")
        source_name = source_path.split("\\")[-1].split("/")[-1]
        sources.append({
            "index": i,
            "name": source_name,
            "snippet": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
        })

    messages = st.session_state.prompt_template.format_messages(
        context=context,
        chat_history=chat_history,
        question=query,
    )

    try:
        response = st.session_state.llm.invoke(messages)
        return response.content, sources
    except Exception:
        if not check_ollama_health():
            st.session_state.ollama_ok = False
            return (
                "⚠️ Ollama 服务在推理过程中断开连接。\n\n"
                "请检查服务状态后点击侧边栏「重新连接」按钮。"
            ), []
        raise


def render_sources(sources: list[dict]):
    """渲染引用来源折叠面板"""
    if not sources:
        return
    with st.expander("📚 引用来源", expanded=False):
        for src in sources:
            st.markdown(
                f"**[资料{src['index']}]** {src['name']}\n\n"
                f"> {src['snippet']}"
            )
            st.divider()


def render_kb_management():
    """渲染知识库管理界面"""
    st.header("知识库管理")

    docs = list_documents()
    st.markdown(f"当前文档数量：**{len(docs)}**")

    if docs:
        with st.expander("查看/删除现有文档", expanded=False):
            for doc_name in docs:
                col1, col2 = st.columns([3, 1])
                col1.text(doc_name)
                if col2.button("删除", key=f"del_{doc_name}"):
                    delete_document(doc_name)
                    st.success(f"已删除: {doc_name}")
                    st.rerun()

    st.subheader("添加文档")
    uploaded_file = st.file_uploader(
        "上传文档（支持 .txt / .pdf / .docx）",
        type=SUPPORTED_EXTENSIONS,
        key="doc_uploader",
    )
    if uploaded_file is not None:
        if st.button("确认添加"):
            file_bytes = uploaded_file.read()
            try:
                extract_text(uploaded_file.name, file_bytes)
                add_document(uploaded_file.name, file_bytes)
                st.success(f"已添加: {uploaded_file.name}")
                st.rerun()
            except Exception as e:
                st.error(f"文件解析失败: {e}")

    st.subheader("重建知识库")
    st.caption("添加或删除文档后，需重建知识库使更改生效")
    if st.button("重新构建知识库"):
        with st.spinner("正在重新构建知识库..."):
            vector_store = rebuild_knowledge_base()
            st.session_state.retriever = Retriever(vector_store)
        st.success("知识库重建完成！")


def render_chat_history_sidebar():
    """渲染对话历史管理侧边栏"""
    st.header("对话历史")

    if st.button("➕ 新建对话"):
        st.session_state.session_id = new_session_id()
        st.session_state.messages = []
        st.session_state.sources = {}
        st.session_state.pop("editing_session", None)
        st.rerun()

    sessions = list_sessions()
    if not sessions:
        st.caption("暂无历史对话")
        return

    for session in sessions[:20]:
        sid = session["id"]
        is_current = sid == st.session_state.session_id

        # 编辑标题模式
        if st.session_state.get("editing_session") == sid:
            new_title = st.text_input(
                "新标题",
                value=session["title"],
                key=f"rename_input_{sid}",
                label_visibility="collapsed",
            )
            c1, c2 = st.columns(2)
            if c1.button("确认", key=f"rename_ok_{sid}"):
                if new_title.strip():
                    rename_session(sid, new_title.strip())
                st.session_state.pop("editing_session", None)
                st.rerun()
            if c2.button("取消", key=f"rename_cancel_{sid}"):
                st.session_state.pop("editing_session", None)
                st.rerun()
            continue

        # 正常显示模式：标题按钮 + 操作图标
        col_title, col_edit, col_export, col_del = st.columns([5, 1, 1, 1])
        label = f"{'▶ ' if is_current else ''}{session['title']}"
        if col_title.button(label, key=f"load_{sid}", disabled=is_current):
            st.session_state.session_id = sid
            st.session_state.messages = load_session(sid)
            st.session_state.sources = {}
            st.rerun()
        if col_edit.button("✏️", key=f"edit_{sid}", help="重命名"):
            st.session_state.editing_session = sid
            st.rerun()
        export_text = export_session_as_text(sid)
        col_export.download_button(
            "📥",
            data=export_text,
            file_name=f"{session['title']}.txt",
            mime="text/plain",
            key=f"export_{sid}",
            help="导出为文本",
        )
        if col_del.button("🗑", key=f"delsess_{sid}", help="删除"):
            delete_session(sid)
            if sid == st.session_state.session_id:
                st.session_state.session_id = new_session_id()
                st.session_state.messages = []
                st.session_state.sources = {}
            st.rerun()


def main():
    st.set_page_config(
        page_title="华智大学智能问答助手",
        page_icon="🎓",
        layout="centered",
    )

    st.title("🎓 华智大学智能问答助手")
    st.caption("我可以帮你解答关于奖学金、选课、研究生培养等校园规章制度方面的问题")

    initialize_system()

    if not st.session_state.ollama_ok:
        st.warning(
            "⚠️ Ollama 服务未就绪，问答功能暂不可用。\n\n"
            "请启动 Ollama 后点击侧边栏「重新连接」按钮。",
            icon="🔌",
        )

    # 渲染已有消息
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                sources = st.session_state.sources.get(str(i), [])
                render_sources(sources)

    # 用户输入
    if query := st.chat_input("请输入你的问题，例如：博士生国家奖学金多少钱？"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response, sources = process_query(query)
            st.markdown(response)
            render_sources(sources)

        msg_index = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.sources[str(msg_index)] = sources

        # 自动保存对话历史
        save_session(st.session_state.session_id, st.session_state.messages)

    # 侧边栏
    with st.sidebar:
        # 对话历史
        render_chat_history_sidebar()

        st.divider()

        # 系统信息
        st.header("系统信息")
        st.markdown(f"**模型:** `{config.LLM_MODEL_NAME}`")
        st.markdown(f"**Embedding:** `{config.EMBEDDING_MODEL_NAME}`")
        st.markdown(f"**检索 Top-K:** `{config.RETRIEVAL_TOP_K}`")

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

        if st.button("清空当前对话"):
            st.session_state.messages = []
            st.session_state.sources = {}
            save_session(st.session_state.session_id, [])
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

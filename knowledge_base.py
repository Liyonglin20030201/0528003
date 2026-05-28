"""模块一：知识库构建

将校园文档拆分成小段并存入 FAISS 向量数据库。
"""
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import config


def load_documents(docs_dir: Path = config.DOCS_DIR) -> list:
    """从指定目录加载所有 txt 文档"""
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"[知识库] 加载了 {len(documents)} 个文档")
    return documents


def split_documents(documents: list) -> list:
    """将文档切分为小片段"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[知识库] 切分为 {len(chunks)} 个文本片段")
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    """获取 Embedding 模型"""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: list, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """构建 FAISS 向量数据库"""
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(config.VECTOR_STORE_DIR))
    print(f"[知识库] 向量数据库已保存至 {config.VECTOR_STORE_DIR}")
    return vector_store


def load_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    """加载已有的向量数据库"""
    return FAISS.load_local(
        str(config.VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def init_knowledge_base() -> FAISS:
    """一键初始化知识库：加载文档 → 切分 → 向量化 → 存储"""
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    vector_store = build_vector_store(chunks, embeddings)
    return vector_store


if __name__ == "__main__":
    init_knowledge_base()
    print("[知识库] 构建完成！")

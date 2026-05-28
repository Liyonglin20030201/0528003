"""模块二：检索模块

根据用户问题，从向量数据库中检索最相关的文档片段。
"""
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

import config
from knowledge_base import get_embeddings, load_vector_store


class Retriever:
    """知识库检索器"""

    def __init__(self, vector_store: FAISS | None = None):
        self.embeddings = get_embeddings()
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = load_vector_store(self.embeddings)

    def retrieve(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[Document]:
        """检索与 query 最相关的 top_k 个文档片段"""
        docs = self.vector_store.similarity_search(query, k=top_k)
        return docs

    def retrieve_with_scores(
        self, query: str, top_k: int = config.RETRIEVAL_TOP_K
    ) -> list[tuple[Document, float]]:
        """检索并返回相似度分数"""
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        return results

from typing import List
from agent.base import MemoryAgent

class Rag:
    def __init__(self, llm, emb, vector_db):
        self.llm = llm
        self.embedding_func = emb
        self.vector_db = vector_db

    def query(self, text: str) -> List[str]:
        # Step 1: 将查询文本转换为嵌入向量
        query_embedding = self.embedding_model(text)

        # Step 2: 从向量数据库检索相关文档
        relevant_docs = self.vector_db.search(query_embedding)

        # Step 3: 使用语言模型生成答案
        context = "\n".join(relevant_docs)

        return context

    def store_vector(self, docs):
        return

    def insert(self):
        pass


class RagAgent(MemoryAgent):
    pass

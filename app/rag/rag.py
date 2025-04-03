import os
from typing import List
from app.agent.base import MemoryAgent
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

from ollama import embed

from app.config import PROJECT_ROOT


class Retriever:
    def __init__(self, config: dict, embed_func: callable, vector_db_name):
        self.vector_db_name = vector_db_name
        self.vector_db_config = config
        self.embed_func = embed_func

    def get_client(self):
        raise NotImplementedError("store doc")

    def store(self, doc: str):
        raise NotImplementedError("store doc")

    def search(self, doc: str):
        raise NotImplementedError("search doc")


class MilvusRetriever(Retriever):
    def __init__(self, **kwargs):
        embed_model = "nomic-embed-text:latest"

        CLUSTER_ENDPOINT = "http://localhost:19530"
        TOKEN = "root:Milvus"

        milvus_config = {"uri": CLUSTER_ENDPOINT, "token": TOKEN}

        def embed_function(doc: str):
            return embed(model=embed_model, input=doc)

        embed_func = embed_function

        super().__init__(milvus_config, embed_func, vector_db_name="milvus")

    def get_client(self) -> MilvusClient:
        client = MilvusClient(**self.vector_db_config)
        return client

    def create_simple_vetcor_collection(self, colection_name):
        client = self.get_client()

        if colection_name in client.list_collections():
            return

        # define fields
        primary_key = FieldSchema(
            name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
        )

        vector = FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=768,
        )

        text = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=256)

        index_params = client.prepare_index_params()

        index_params.add_index(field_name="id", index_type="AUTOINDEX")

        index_params.add_index(
            field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
        )

        schema = CollectionSchema(fields=[primary_key, vector, text])
        client.create_collection(
            collection_name=colection_name, schema=schema, index_params=index_params
        )
        client.close()

    def store(self, doc, collection_name):
        resp = self.embed_func(doc)
        client = self.get_client()
        data = resp.embeddings[0]

        insert_ans = client.insert(
            collection_name=collection_name, data={"vector": data, "text": doc}
        )
        print(insert_ans)

    def search(self, doc, collection_name):
        resp = self.embed_func(doc)
        client = self.get_client()
        data = resp.embeddings[0]
        search_result = client.search(
            collection_name=collection_name,
            data=[data],
            limit=2,
            output_fields=["*"],
            search_params={"metric_type": "COSINE"},  # 指定度量类型
            anns_field="vector",
        )
        return [result[0]["entity"]["text"] for result in search_result]


class RagAgent(MemoryAgent):
    def load_knowledge_lib(self, path, collection_name):
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file at path {path} does not exist.")
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()

        # Split content by empty lines
        documents = [doc.strip() for doc in content.split("\n\n") if doc.strip()]

        self.retriever.create_simple_vetcor_collection(self.collection_name)

        # Store each document in the vector database
        for doc in documents:
            self.retriever.store(doc, collection_name)

    def __init__(self, name, desc, llm, max_message):
        super().__init__(name, desc, llm, max_message)
        self.retriever = MilvusRetriever()

        self.collection_name = "default_rag_collection"

        txt_path = PROJECT_ROOT / "app" / "rag" / "all.txt"


        self.load_knowledge_lib(path=txt_path, collection_name=self.collection_name)

    def make_prompt(self, content: List[str], q: str):
        info = ""
        for x in content:
            info += x
            info += "/n"
        return f"""你有以下内容可以参考{info}\n这些额外的信息是提供给你的，请你决定参考的程度，然后回答用户的问题:{q} """

    async def run(self, mode="debug"):
        while True:
            q = input("请输入您的问题(quit 或者 exit退出):\nUser:")
            if q == "quit" or q == "exit":
                return

            print("正在检索知识库...")
            content = self.retriever.search(q, collection_name=self.collection_name)

            prompt = self.make_prompt(content, q)

            self.update_memory(prompt, "user")

            resp = await self.llm.ask(self.memory.messages)  # 上下文的来源，token消耗
            print("Agent:", resp, "\n")
            self.update_memory(resp)


def create_test_doc():
    txt_path = PROJECT_ROOT / "app" / "rag" / "all.txt"
    if txt_path.exists():
        return
    with open(txt_path, "w", encoding="utf8") as f:
        f.writelines(
            [
                "邪恶番茄意面是由煤气罐博士发明出来的美味食物，在俱乐部中，呆头和大王都很爱吃这道菜。\n\n",
                "邪恶番茄意面使用了 森林小镇最新鲜的番茄，这种番茄是甜的。\n\n",
                "有一次呆头和大王把这道菜分享给刚来此地的小汪，小汪品尝之后说 相见恨晚！\n\n",
            ]
        )
    return


def delete_test_doc():
    txt_path = PROJECT_ROOT / "app" / "rag" / "all.txt"
    if txt_path.exists():
        os.remove(txt_path)

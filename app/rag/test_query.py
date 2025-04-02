import time
import ollama
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
import numpy as np
import sys

# 运行这个代码的时候，不能使用代理，否则本应请求的11434会转发到代理端口
docs = [
    "刘禹锡 772年-842年 字梦得 荥阳（今河南省郑州市荥阳市）人，祖籍洛阳（今河南省洛阳市.",
    "刘禹锡诗文俱佳，涉猎题材广泛，与白居易并称“刘白”，与柳宗元并称“刘柳”，与韦应物、白居易合称“三杰”，留有《陋室铭》《竹枝词》《杨柳枝词》《乌衣巷》等名篇。",
    "刘禹锡的山水诗，改变了大历、贞元诗人襟幅狭小、气象萧瑟的风格，而常常是写一种超出空间实距的、半虚半实的开阔景象",
]


CLUSTER_ENDPOINT = "http://localhost:19530"
TOKEN = "root:Milvus"

# 1. Set up a Milvus client
client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)

print("collections", client.list_collections())

name = "nomic_emb_text_test"
if name in client.list_collections():
    client.drop_collection(collection_name=name)


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
client.create_collection(collection_name=name, schema=schema, index_params=index_params)

client.load_collection(collection_name=name)

for i, doc in enumerate(docs):
    resp = ollama.embed(
        model="nomic-embed-text:latest",
        input=doc,
    )
    data = resp.embeddings[0]

    insert_ans = client.insert(collection_name=name, data={"vector": data, "text": doc})
    print(insert_ans)

time.sleep(5)

res = client.query(
    collection_name=name, output_fields=["id", "text"], limit=10, filter=""
)
print(res)

# sys.exit(1)

q = "刘禹锡和白居易之间有什么关系？"

resp = ollama.embed(
    model="nomic-embed-text:latest",
    input=q,
)

e_embdding = resp.embeddings[0]

search_result = client.search(
    collection_name=name,
    data=[e_embdding],
    limit=2,
    output_fields=["*"],
    search_params={"metric_type": "COSINE"},  # 指定度量类型
    anns_field="vector",
)

for result in search_result:
    print("\n\n搜索结果->", result[0]["entity"]["text"])

import ollama
from pymilvus import MilvusClient
import numpy as np

# 运行这个代码的时候，不能使用代理，否则本应请求的11434会转发到代理端口

input = "This is a test input sentence."

resp = ollama.embed(
    model="nomic-embed-text:latest",
    input=input,
)

print(resp.embeddings)

print(len(resp.embeddings))

print(len(resp.embeddings[0]))  # 768


CLUSTER_ENDPOINT = "http://localhost:19530"
TOKEN = "root:Milvus"

# 1. Set up a Milvus client
client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)


name = "nomic_emb_text_test"
if name in client.list_collections():
    client.drop_collection(collection_name=name)

client.create_collection(collection_name=name, dimension=768)

data = np.array(resp.embeddings[0], dtype=np.float32)

insert_ans = client.insert(collection_name=name, data={"id":1, "vector":data, "text": input})
print(insert_ans)

res = client.query(collection_name=name, output_fields=["id", "vector", "text"], limit=10)
print(res)



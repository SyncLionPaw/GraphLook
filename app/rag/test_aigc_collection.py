from pymilvus import MilvusClient, DataType
from pprint import pprint

CLUSTER_ENDPOINT = "http://localhost:19530"
TOKEN = "root:Milvus"

# 1. Set up a Milvus client
client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN 
)

if "quick_setup" not in client.list_collections():

    # 2. Create a collection in quick setup mode
    client.create_collection(
        collection_name="quick_setup",
        dimension=5
    )

res = client.get_load_state(
    collection_name="quick_setup"
)

print(res)
cs = client.list_collections()
print(cs)

info = client.describe_collection(collection_name="quick_setup")
pprint(info)

client.drop_collection(collection_name="quick_setup")

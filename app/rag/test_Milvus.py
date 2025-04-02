from pymilvus import DataType, MilvusClient
from pymilvus import model


client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")


databases = client.list_databases()

DATABASE = "my_database_1"

if DATABASE not in databases:
    client.create_database(DATABASE)

if "c1" not in client.list_collections():

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True
    )

    # 3.3. Prepare index parameters
    index_params = client.prepare_index_params()

    # 3.4. Add indexes
    index_params.add_index(
        field_name="my_id",
        index_type="AUTOINDEX"
    )

    index_params.add_index(
        field_name="my_vector", 
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    # 3.2. Add fields to schema
    schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)
    schema.add_field(field_name="my_varchar", datatype=DataType.VARCHAR, max_length=512)

    client.create_collection(collection_name="c1", schema=schema, index_params=index_params)

res = client.get_load_state(collection_name="c1")

print(res)
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# 1️⃣ 连接 Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)


collection_name = "rag_qwen_v4"
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=10000),
    FieldSchema(name="coarse_types", dtype=DataType.JSON),
    FieldSchema(name="entities", dtype=DataType.JSON),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4096),
    FieldSchema(name="coarse_type_set", dtype=DataType.VARCHAR, max_length=2000)  # 新增字段
]

schema = CollectionSchema(fields, description="NER training data with coarse_type_set")
new_collection = Collection(collection_name, schema=schema)
print(f'Collection {collection_name} created')
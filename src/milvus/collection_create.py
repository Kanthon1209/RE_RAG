from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# 1️⃣ 连接 Milvus
connections.connect(
    alias="default",
    host="localhost",  # Milvus 服务地址
    port="19530"       # Milvus 默认端口
)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="coarse_types", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="input", dtype=DataType.VARCHAR, max_length=5000),
    FieldSchema(name="instruction", dtype=DataType.VARCHAR, max_length=5000),
    FieldSchema(name="output", dtype=DataType.VARCHAR, max_length=5000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4096) # Qwen3-embedding 是 4096 维度
]

schema = CollectionSchema(fields, description="NER/RE training samples")

# 2️⃣ 定义 Collection 名称
collection_name = "rag_qwen"

# 3️⃣ 检查 Collection 是否存在
available_collections = utility.list_collections()
print(available_collections)
if collection_name in available_collections:
    print(f"Collection '{collection_name}' 已存在")
    collection = Collection(collection_name)  # 可以直接使用已有 
else:
    print(f"Collection '{collection_name}' 不存在，正在创建...")
    collection = Collection(name=collection_name, schema=schema) # 创建
    print(f"Collection '{collection_name}' 创建成功")




if collection_name in available_collections:
    print(f"Collection '{collection_name}' 已存在")
    collection = Collection(collection_name)  # 可以直接使用已有 collection
else:
    print(f"Collection '{collection_name}' 不存在，正在创建...")
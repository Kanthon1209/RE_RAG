from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# 1️⃣ 连接 Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# 2️⃣ 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 自动ID, 所以插入的时候不要传入 ids
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=10000),
    FieldSchema(name="coarse_types", dtype=DataType.JSON, max_length=2000),   # 可以存 JSON 字符串化后的数组
    FieldSchema(name="entities", dtype=DataType.JSON, max_length=5000),       # 同样存 JSON 字符串化后的对象数组
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4096)         # Qwen3 embedding 输出维度
]

schema = CollectionSchema(fields, description="NER training data (new format)")

# 3️⃣ Collection 名称
collection_name = "rag_qwen_json"
# 4️⃣ 检查 Collection 是否存在
available_collections = utility.list_collections()

if collection_name in available_collections:
    print(f"✅ Collection '{collection_name}' 已存在")
    collection = Collection(collection_name)
else:
    print(f"⚙️ Collection '{collection_name}' 不存在，正在创建...")
    collection = Collection(name=collection_name, schema=schema)
    print(f"✅ Collection '{collection_name}' 创建成功")

from pymilvus import connections, Collection, FieldSchema, DataType
import json

# ===========================
# 1️⃣ 连接 Milvus
# ===========================
connections.connect(alias="default", host="localhost", port="19530")
collection_name = "rag_qwen"
collection = Collection(collection_name)

# ===========================
# 2️⃣ 检查并新增字段 coarse_type_set
# ===========================
existing_fields = [f.name for f in collection.schema.fields]
if "coarse_type_set" not in existing_fields:
    collection.add_field(FieldSchema(name="coarse_type_set", dtype=DataType.VARCHAR, max_length=2000))
    print("✅ 字段 coarse_type_set 已添加")

# ===========================
# 3️⃣ 批量生成 coarse_type_set
# ===========================
batch_size = 1000
results = collection.query(output_fields=["id", "entities"])
total = len(results)
print(f"总记录数: {total}")

for i in range(0, total, batch_size):
    batch = results[i:i+batch_size]
    updates = []
    for r in batch:
        entities = r["entities"]
        if isinstance(entities, str):
            entities = json.loads(entities)
        type_set = set(ent["coarse_type"] for ent in entities)
        updates.append({"id": r["id"], "coarse_type_set": ",".join(type_set)})
    collection.update(updates)
    print(f"✅ 更新 {i}~{i+len(batch)} 条记录的 coarse_type_set")
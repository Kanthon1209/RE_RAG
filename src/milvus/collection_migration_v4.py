from pymilvus import connections, Collection
import json

# 1️⃣ 连接 Milvus
connections.connect(alias="default", host="localhost", port="19530")

old_col = Collection("rag_qwen_origin")
new_col = Collection("rag_qwen_v4")  # 新 collection，已包含额外字段

batch_size = 200
migrated = 0

# 2️⃣ 创建 query_iterator（全量查询）
iterator = old_col.query_iterator(
    output_fields=["source", "sentence", "coarse_types", "entities", "embedding"],
    batch_size=batch_size
)

# 3️⃣ 正确迭代 QueryIterator
while True:
    try:
        batch = iterator.next()  # 获取下一批
    except StopIteration:
        break  # 遍历结束

    if not batch:
        break

    batch_source = []
    batch_sentence = []
    batch_coarse = []
    batch_entities = []
    batch_embeds = []
    batch_type_set = []

    for item in batch:
        entities = json.loads(item["entities"]) if isinstance(item["entities"], str) else item["entities"]
        coarse_types = json.loads(item["coarse_types"]) if isinstance(item["coarse_types"], str) else item["coarse_types"]

        # 生成 coarse_type_set
        type_set_str = ",".join(set(ent["coarse_type"] for ent in entities if "coarse_type" in ent))

        batch_source.append(item["source"])
        batch_sentence.append(item["sentence"])
        batch_coarse.append(coarse_types)
        batch_entities.append(entities)
        batch_embeds.append(item["embedding"])
        batch_type_set.append(type_set_str)

    # 4️⃣ 插入新 Collection
    new_col.insert([
        batch_source,
        batch_sentence,
        batch_coarse,
        batch_entities,
        batch_embeds,
        batch_type_set
    ])

    migrated += len(batch)
    print(f"✅ 已迁移 {migrated} 条记录")

iterator.close()
print(f"🎉 数据迁移完成，总记录数: {migrated}")

from pymilvus import connections, Collection
import json
from tqdm import tqdm

# 1️⃣ Connect to Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# 2️⃣ Define source and target collections
old_name = "rag_qwen"
new_name = "rag_qwen_json"

old_col = Collection(old_name)
new_col = Collection(new_name)

# 3️⃣ Create a query iterator (safe pagination)
iterator = old_col.query_iterator(
    batch_size=200,  # You can tune this (100~500 typical)
    output_fields=["source", "sentence", "coarse_types", "entities", "embedding"]
)

migrated = 0

# 4️⃣ Iterate through all batches
while True:
    batch = iterator.next()
    if not batch:
        break

    batch_source = []
    batch_sentence = []
    batch_coarse = []
    batch_entities = []
    batch_embeds = []

    for item in batch:
        batch_source.append(item["source"])
        batch_sentence.append(item["sentence"])

        # Safely parse string → JSON
        def parse_json(value):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return []  # fallback empty list
            return value

        batch_coarse.append(parse_json(item["coarse_types"]))
        batch_entities.append(parse_json(item["entities"]))
        batch_embeds.append(item["embedding"])

    # Insert into new collection
    new_col.insert([
        batch_source,
        batch_sentence,
        batch_coarse,
        batch_entities,
        batch_embeds
    ])

    migrated += len(batch)
    print(f"✅ Migrated {migrated} records so far...")

iterator.close()

print(f"🎉 Migration complete. Total migrated: {migrated}")

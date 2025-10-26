import json
import os
from tqdm import tqdm
from pymilvus import connections, Collection
from src.qwen.remote_services import Client
import debugpy

# # ===== Debug 监听 =====
# debugpy.listen(("localhost", 6666))
# print("waiting for debugger...")
# debugpy.wait_for_client()

# ===== 1️⃣ 连接到 Milvus =====
connections.connect("default", host="127.0.0.1", port="19530")
collection = Collection("rag_qwen_origin")

# ===== 2️⃣ 加载数据 =====
with open("data/train_origin.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print("data length:", len(data))

# ===== 3️⃣ 断点与错误批次文件 =====
checkpoint_path = "progress.json"
error_batches_path = "error_batches.json"

# 加载断点
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)
        start_index = checkpoint.get("last_index", 0)
else:
    start_index = 0

# 加载出错批次
if os.path.exists(error_batches_path):
    with open(error_batches_path, "r") as f:
        error_batches = set(json.load(f))
else:
    error_batches = set()

print(f"▶️ 从第 {start_index} 条开始处理")
print(f"▶️ 已有出错批次: {sorted(error_batches)}")

# ===== 4️⃣ 批次大小 =====
batch_size = 100  # 每批 100 条

# ===== 主循环 =====
with Client() as client:
    for i in range(start_index, len(data), batch_size):
        batch_num = i // batch_size
        if batch_num in error_batches:
            print(f"⏩ 跳过已出错批次 {batch_num}")
            continue

        batch = data[i:i + batch_size]

        ids, sources, sentences, coarse_types_list, entities_list = [], [], [], [], []
        for j, item in enumerate(batch):
            idx = i + j
            sentence = item.get("sentence", "").strip()
            if not sentence:
                continue
            ids.append(idx)
            sources.append(item.get("source", ""))
            sentences.append(sentence)
            coarse_types_list.append(item.get("coarse_types", []))
            entities_list.append(item.get("entities", []))

        try:
            # 🚀 一次请求多个文本
            resp = client.embed(sentences)
            embeddings = resp["embeddings"]

            # ===== 插入 Milvus =====
            collection.insert([
                sources,
                sentences,
                coarse_types_list,
                entities_list,
                embeddings
            ])
            print(f"✅ 成功插入 {len(embeddings)} 条 (索引范围 {i}-{i + len(embeddings) - 1})")

            # 每隔几批再 flush 一次，提高写入效率
            if batch_num % 10 == 0:
                collection.flush()

            # ===== 更新断点 =====
            with open(checkpoint_path, "w") as f:
                json.dump({"last_index": i + batch_size}, f)
        except Exception as e:
            print(f"❌ 第 {batch_num} 批处理失败: {e}")
            error_batches.add(batch_num)
            with open(error_batches_path, "w") as f:
                json.dump(sorted(list(error_batches)), f)


print("\n🎉 全部处理完成！")
print(f"⚠️ 还有出错批次未处理: {sorted(error_batches)}")

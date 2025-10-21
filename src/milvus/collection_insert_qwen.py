import json
import os
from tqdm import tqdm
from openai import OpenAI
from pymilvus import connections, Collection
from src.qwen.embedding import EmbeddingClient

# ===== 1️⃣ 连接到 Milvus =====
connections.connect("default", host="127.0.0.1", port="19530")
collection = Collection("rag_qwen")

# ===== 2️⃣ 加载数据 =====
with open("data/train_v3.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    print("data length:", len(data))

# ===== 4️⃣ 设置断点文件 =====
checkpoint_path = "progress.json"
error_batches_path = "error_batches.json"

# 加载正常进度
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)
        start_index = checkpoint.get("last_index", 0)
else:
    start_index = 0

# 加载出错批次列表
if os.path.exists(error_batches_path):
    with open(error_batches_path, "r") as f:
        error_batches = set(json.load(f))
else:
    error_batches = set()

print(f"▶️ 从第 {start_index} 条开始处理")
print(f"▶️ 已有出错批次: {sorted(error_batches)}")

# ===== 5️⃣ 设置批大小 =====
batch_size = 100  # 每批处理 100 条（可调）

# ===== 6️⃣ 主循环 =====
for i in range(start_index, len(data), batch_size):
    batch_num = i // batch_size
    if batch_num in error_batches:
        print(f"⏩ 跳过已出错批次 {batch_num} (索引 {i}~{i+batch_size-1})")
        continue

    batch = data[i:i + batch_size]

    ids, sources, coarse_types_list, inputs, instructions, outputs, embeddings = [], [], [], [], [], [], []

    print(f"\n🚀 正在处理第 {batch_num + 1} 批 (索引 {i} ~ {i+len(batch)-1})")

    try:
        with EmbeddingClient() as client:
            for j, item in enumerate(tqdm(batch)):
                idx = i + j
                if "output" not in item:
                    continue

                ids.append(idx)
                sources.append(item.get("source", ""))
                coarse_types_list.append(",".join(item.get("coarse_types", [])))
                inputs.append(item["input"])
                instructions.append(item.get("instruction", ""))
                outputs.append(item["output"])

                # ===== 调用 embedding API =====
                resp = client.embed(texts=item["input"])
                embeddings.append(resp.embeddings[0])

        # ===== 插入到 Milvus =====
        if embeddings:
            collection.insert([
                ids, sources, coarse_types_list, inputs, instructions, outputs, embeddings
            ])
            collection.flush()
            print(f"✅ 成功插入 {len(embeddings)} 条 (索引范围 {i}-{i + len(embeddings) - 1})")
        else:
            print(f"⚠️ 本批无可插入数据，跳过")

        # ===== 更新断点 =====
        with open(checkpoint_path, "w") as f:
            json.dump({"last_index": i + batch_size}, f)
        print(f"💾 断点已保存 -> last_index = {i + batch_size}")

    except Exception as e:
        print(f"❌ 第 {batch_num} 批处理失败: {e}")
        error_batches.add(batch_num)
        # 保存出错批次
        with open(error_batches_path, "w") as f:
            json.dump(sorted(list(error_batches)), f)
        print(f"⚠️ 已记录出错批次 {batch_num}, 下一次可以重新处理")
        continue  # 跳过该批次，继续下一批

print("\n🎉 全部处理完成！")
print(f"⚠️ 还有出错批次未处理: {sorted(error_batches)}")

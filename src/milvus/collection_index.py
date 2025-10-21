from pymilvus import connections, Collection


connections.connect(alias="default", host="localhost", port="19530")
collection = Collection("rag_qwen")

index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 256}  # 桶数可以小一些，数据量小，nlist 不要太大
}

collection.create_index("embedding", index_params)
collection.load()
print("✅ IVF_FLAT 索引创建完成")
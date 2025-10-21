from pymilvus import connections, Collection


connections.connect(alias="default", host="localhost", port="19530")
collection = Collection("rag_qwen")
collection.release() # 先释放

# 先检查索引是否存在
indexes = collection.indexes
if indexes:
    collection.drop_index()
    print("✅ 索引已删除")
else:
    print("⚠️ 该字段没有索引")
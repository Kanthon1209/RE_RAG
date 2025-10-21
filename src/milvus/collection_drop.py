from pymilvus import connections, Collection, utility

connections.connect(alias="default", host="localhost", port="19530")
collection_name = "rag_qwen"
collection = Collection(collection_name)
utility.list_collections()
utility.drop_collection(collection_name)
print(f"Collection '{collection_name}' 已删除")
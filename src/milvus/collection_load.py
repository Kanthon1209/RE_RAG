from pymilvus import connections, Collection

connections.connect(alias="default", host="localhost", port="19530")
collection = Collection("rag_qwen")
collection.load()
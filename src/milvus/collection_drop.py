from pymilvus import connections, Collection, utility

connections.connect(alias="default", host="localhost", port="19530")
print("Collections: ", utility.list_collections())
collection_name = input("Input Collection Name: ")
collection = Collection(collection_name)
yes = input("RU Sure?[yes/no]")
if yes == 'yes':
    utility.drop_collection(collection_name)
    print(f"Collection '{collection_name}' 已删除")
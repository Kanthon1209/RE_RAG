from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pymilvus import connections, Collection
import logging
from src.utils.misc import timer
import time
import json
from src.qwen.embedding import EmbeddingClient



collection_name = "rag_qwen"
global_counter = 0

app = FastAPI()
logger = logging.getLogger("app")
client = EmbeddingClient()

# ===== 1️⃣ 初始化 Milvus 连接 =====
connections.connect("default", host="127.0.0.1", port="19530")
collection = Collection(collection_name)
collection.load() # 加载集合到内存

@timer(logger=logger)
def create_embedding(query_text: str) -> list[float]:
    
    resp = client.embed(query_text)
    query_vector = resp['embeddings'][0]
    return query_vector

def do_search():
    ...
    

# ===== 3️⃣ 定义 POST 接口 =====
@app.post("/search")
async def search_items(request: Request):
    start_time = time.time()
    global global_counter
    global_counter += 1

    body = await request.json()

    # 提取请求字段
    query_text = body.get("sentence", None)
    limit = body.get("limit", 10)
    test_coarse_types: list[str] = body.get("coarse_types", [])
    filter_triplets_flag: bool = body.get("filter_triplets_flag", False)
    if filter_triplets_flag not in [True, False]: # 参数校验
        return JSONResponse({"error": "Wrong *filter_triplets_flag* field"}, status_code=400)
    # TODO: 以测试样本 coarse_types 字段为依据, 每一个 type 从训练集中找一个符合的训练样本供给参考


    logger.info(f"{global_counter:<5} received POST /search from {request.client.host}:{request.client.port}")
    output_fields = body.get("output_fields", ["id", "source", "coarse_types", "sentence", "entities"])
    #TODO: 参数校验

    if not query_text:
        return JSONResponse({"error": "missing 'sentence' field"}, status_code=400)

    # ===== 生成 embedding =====
    try:
        query_vector = create_embedding(query_text)
    except Exception as e:
        return JSONResponse({"error": f"embedding generation failed: {e}"}, status_code=500)
    
    # ===== 执行 Milvus 向量检索 =====
    search_params = {"nprobe": 10}
    try:
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=output_fields
        )
    except Exception as e:
        return JSONResponse({"error": f"milvus search failed: {e}"}, status_code=500)

    # ===== 整理返回结果 =====
    hits_list = []
    for hits in results:
        for hit in hits:
            d = {}
            entity = hit.entity
            d["score"] = hit.distance
            for field in output_fields:
                d[field] = entity.get(field)
            # if "coarse_types" in d: # NOTE: 将 coarse_types 字符串转换为列表
            #     d["coarse_types"] = d["coarse_types"].split(",")
            if "coarse_types" in d:
                d["coarse_types"] = json.loads(d["coarse_types"])
            if "entities" in d:
                d["entities"] = json.loads(d["entities"])

            hits_list.append(d)

    end_time = time.time()
    return JSONResponse(
            {
                "results": hits_list, 
                "help": 
                """1. Choosable return fields: ...\n2. default limit value is 10\n3. use coarse_types filed to specify the range of the recognition\n4. 训练样本中, 返回的三元组中的粗粒度标签可能不在要求的 coarse_types 中, 如果需要过滤这一部分的三元组, 避免他们出现在 output 中, 指定 filter_coarse_type 字段为 true, 注意, 不要使用引号包裹 true; filter_coarse_type 不指定值的时候, 等同于设置为 false\n5. """,
                "info": "version 1.0",
                "cost_time": f"{end_time - start_time:.4f}s"
            }, status_code=200)

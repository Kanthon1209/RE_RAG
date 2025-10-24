from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pymilvus import connections, Collection


import logging
import time
import json

from src.qwen.embedding import EmbeddingClient
from src.utils.misc import timer
from src.services.postprocess.rerank import compute_entity_name_match_score, compute_entity_overlap_score

collection_name = "rag_qwen"            # 没有用 JSON 字段的 数据库 (过滤过的)
collection_v2_name = "rag_qwen_json"    # 用了 JSON 字段的数据库 (过滤过的)
collection_v3_name = "rag_qwen_origin"  # 用了 JSON 字段的数据库, 没有经过过滤, 用的原始的训练样本
collection_v4_name = "rag_qwen_v4"  # 用了 JSON 字段的数据库, 没有经过过滤, 用的原始的训练样本
global_counter = 0

app = FastAPI()
logger = logging.getLogger("app")
client = EmbeddingClient()

# ===== 1️⃣ 初始化 Milvus 连接 =====
connections.connect("default", host="127.0.0.1", port="19530")
collection = Collection(collection_name)
collection.load() # 加载集合到内存
collection_v2 = Collection(collection_v2_name)
collection_v2.load() # 加载集合到内存
collection_v3 = Collection(collection_v3_name)
collection_v3.load() # 加载集合到内存
collection_v4 = Collection(collection_v4_name)
collection_v4.load() # 加载集合到内存

@timer(logger=logger)
def create_embedding(query_text: str) -> list[float]:
    
    resp = client.embed(query_text)
    query_vector = resp['embeddings'][0]
    return query_vector




# ===== 3️⃣ 定义 POST 接口 =====
@app.post("/v1/search")
async def search_items(request: Request):
    start_time = time.time()
    global global_counter
    global_counter += 1

    body = await request.json()

    # 提取请求字段
    query_text = body.get("sentence", None)
    limit = body.get("limit", 50)
    test_coarse_types: list[str] = body.get("coarse_types", [])
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
    search_params = {
        "metric_type": "IP",  # 必须和 index 一致, 改成 Inner Product 内积
        "params": {"ef": 128} # NOTE: ef: 查询时的搜索范围，越大精度越高、但速度略慢
    }
    
    # 从服务器获取 embedding 的时候, 服务器上做了 L2 归一化, 这里 内积（Inner Product, IP） 和 余弦相似度（Cosine similarity）是等价的
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
                if test_coarse_types:
                    overlap_score = compute_entity_overlap_score(test_coarse_types, d["entities"])
                    d['coarse_types_overlap_score'] = overlap_score
                else:
                    d['coarse_types_overlap_score'] = 0
                entity_name_match_score = compute_entity_name_match_score(query_text, d["entities"])
                d['entity_name_match_score'] = entity_name_match_score
                d['mix_score'] = (d['entity_name_match_score'] + 0.1) * (d['coarse_types_overlap_score'] + 0.1)
            hits_list.append(d)

    # 根据度量类型来
    hits_list.sort(key=lambda x: (-x['mix_score'])) # 按照 overlap_score 先排序, 后 score 排序
    end_time = time.time()
    return JSONResponse(
            {
                "results": hits_list, 
                "help": 
                """1. Choosable return fields: ...\n2. default limit value is 10\n3. use coarse_types filed to specify the range of the recognition\n4. 训练样本中, 返回的三元组中的粗粒度标签可能不在要求的 coarse_types 中, 如果需要过滤这一部分的三元组, 避免他们出现在 output 中, 指定 filter_coarse_type 字段为 true, 注意, 不要使用引号包裹 true; filter_coarse_type 不指定值的时候, 等同于设置为 false\n5. """,
                "info": "version 1.0",
                "cost_time": f"{end_time - start_time:.4f}s"
            }, status_code=200)


@app.post("/v2/search")
async def search_items_2(request: Request):
    start_time = time.time()
    global global_counter
    global_counter += 1

    body = await request.json()

    # 提取请求字段
    query_text = body.get("sentence", None)
    limit = body.get("limit", 50)
    test_coarse_types: list[str] = body.get("coarse_types", [])
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
    search_params = {
        "metric_type": "IP",  # 必须和 index 一致, 改成 Inner Product 内积
        "params": {"ef": 128} # NOTE: ef: 查询时的搜索范围，越大精度越高、但速度略慢
    }
    
    # 从服务器获取 embedding 的时候, 服务器上做了 L2 归一化, 这里 内积（Inner Product, IP） 和 余弦相似度（Cosine similarity）是等价的
    try:
        results = collection_v2.search(
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
            # NOTE:
            if test_coarse_types:
                overlap_score = compute_entity_overlap_score(test_coarse_types, d["entities"])
                d['coarse_types_overlap_score'] = overlap_score
            else:
                d['coarse_types_overlap_score'] = 0
            entity_name_match_score = compute_entity_name_match_score(query_text, d["entities"])
            d['entity_name_match_score'] = entity_name_match_score
            d['mix_score'] = (d['entity_name_match_score'] + 0.1) * (d['coarse_types_overlap_score'] + 0.1)
            # NOTE:
            hits_list.append(d)

    # 根据度量类型来
    hits_list.sort(key=lambda x: (-x['mix_score'])) # 按照 overlap_score 先排序, 后 score 排序
    end_time = time.time()
    return JSONResponse(
            {
                "results": hits_list, 
                "info": "version 2.0",
                "cost_time": f"{end_time - start_time:.4f}s"
            }, status_code=200)



@app.post("/v3/search")
async def search_v3(request: Request):
    start_time = time.time()
    global global_counter
    global_counter += 1

    body = await request.json()

    # 提取请求字段
    query_text = body.get("sentence", None)
    limit = body.get("limit", 50)
    methods = ['default', "mix"]
    method = body.get("method", "default")
    if method not in methods:
        return JSONResponse({"error": f"wrong method, method should be in {methods}"}, status_code=400)

    test_coarse_types: list[str] = body.get("coarse_types", [])
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
    search_params = {
        "metric_type": "IP",  # 必须和 index 一致, 改成 Inner Product 内积
        "params": {"ef": 128} # NOTE: ef: 查询时的搜索范围，越大精度越高、但速度略慢
    }
    
    try:
        results = collection_v3.search(
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
            # NOTE:
            if test_coarse_types:
                overlap_score = compute_entity_overlap_score(test_coarse_types, d["entities"])
                d['coarse_types_overlap_score'] = overlap_score
            else:
                d['coarse_types_overlap_score'] = 0
            entity_name_match_score = compute_entity_name_match_score(query_text, d["entities"])
            d['entity_name_match_score'] = entity_name_match_score
            d['mix_score'] = (d['entity_name_match_score'] + 0.1) * (d['coarse_types_overlap_score'] + 0.1)
            # NOTE:
            hits_list.append(d)

    # 根据度量类型来
    if method == 'mix':
        hits_list.sort(key=lambda x: (-x['mix_score'])) # 按照 overlap_score 先排序, 后 score 排序
    end_time = time.time()
    return JSONResponse(
            {
                "results": hits_list, 
                "info": "version 2.0",
                "cost_time": f"{end_time - start_time:.4f}s"
            }, status_code=200)


def select_train_samples_by_coarse_type(coarse_types: list[str]) -> list[dict]:
    expr_parts = [f'coarse_type_set like "%{ctype}%"' for ctype in coarse_types]
    expr = " or ".join(expr_parts)
    filtered_results: list = collection_v4.query(
        expr=expr,
        output_fields=["id", "entities", "coarse_type_set", "sentence"]
    )
    for item in filtered_results:
        item_coarse_count = 0
        for cs_type in item['coarse_type_set'].split(','):
            if cs_type in coarse_types:
                item_coarse_count += 1
        item['coarse_count'] = item_coarse_count

    filtered_results.sort(key= lambda item: -item['coarse_count'])

    rt_list = []
    for datum in filtered_results:
        if len(coarse_types) == 0:
            break
        modified_flag = False
        for c_type in datum['coarse_type_set'].split(','):
            if c_type in coarse_types:
                coarse_types.remove(c_type)
                modified_flag = True
        if modified_flag:
            rt_list.append(datum)
    return JSONResponse({"data": rt_list})

# ======================
# 5️⃣ API 接口
# ======================
@app.post("/select_samples")
async def api_select_samples(request: Request):
    body = await request.json()
    test_coarse_types = body.get("coarse_type", None)
    if test_coarse_types is None:
        return JSONResponse({"error": "no test sample coarse_type"})
    return select_train_samples_by_coarse_type(test_coarse_types)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pymilvus import connections, Collection


import logging
import time
import json

from src.qwen.remote_services import Client
from src.utils.misc import timer
from src.services.postprocess.rerank import compute_entity_name_match_score, compute_entity_overlap_score

collection_name = "rag_qwen"            # 没用 JSON 字段的 数据库 (过滤过的)
collection_v2_name = "rag_qwen_json"    # 用了 JSON 字段的数据库 (过滤过的)
collection_v3_name = "rag_qwen_origin"  # 用了 JSON 字段的数据库, 没有经过过滤, 用的原始的训练样本
collection_v4_name = "rag_qwen_v4"      # 用了 JSON 字段的数据库, 没有经过过滤, 用的原始的训练样本
global_counter = 0

app = FastAPI()
logger = logging.getLogger("app")
client = Client()

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

def rerank(query: str, docs: list[str], model_name: str = "Qwen3-Reranker-4B") -> list[float]:
    return client.rerank(query=query, docs=docs, model_name=model_name)

@app.post("/v3/search")
async def search_v3(request: Request):
    start_time = time.time()
    global global_counter
    global_counter += 1

    body = await request.json()

    # 提取请求字段
    query_text = body.get("sentence", None)
    limit = body.get("limit", 50)
    methods = ["default", "coarse_only", "type_first", "name_first", "mix", "rerank", "rerank-0.6b", "rerank-4b", "rerank-8b"]
    method = body.get("method", "default")
    test_coarse_types: list[str] = body.get("coarse_type", [])


    # 判断使用的方法
    if method not in methods: # 选择的方法不在 定义好的 methods 中的话
        return JSONResponse({"error": f"wrong method, method should be in {methods}"}, status_code=400)
    

    # TODO: 以测试样本 coarse_types 字段为依据, 从训练集中找出若干样本, 这些样本的 entity.coarse_type 可以覆盖测试样本的 coarse_type
    if method == 'coarse_only':
        if test_coarse_types is None or len(test_coarse_types) == 0:
            return JSONResponse({"error": "no test sample coarse_type"})
        return select_train_samples_by_coarse_type(test_coarse_types)

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
                d['type_overlap_score'] = overlap_score
            else:
                d['type_overlap_score'] = 0
            entity_name_match_score = compute_entity_name_match_score(query_text, d["entities"])
            d['name_overlap_score'] = entity_name_match_score
            d['mix_score'] = (d['name_overlap_score'] + 0.1) * (d['type_overlap_score'] + 0.1)
            # NOTE:
            hits_list.append(d)

    # 根据度量类型来
    if method == 'mix': #如果是 mix 重排策略, 将 hits_list 重新排序
        hits_list.sort(key=lambda x: (-x['mix_score'])) # 按照 overlap_score 先排序, 后 score 排序
    if method == 'type_first':
        hits_list.sort(key = lambda x: (-x['type_overlap_score']))
    if method == 'name_first':
        hits_list.sort(key = lambda x: (-x['name_overlap_score']))
    if method in ["rerank", "rerank-0.6b", "rerank-4b", "rerank-8b"]:
        method_model_map = {
            "rerank": "Qwen3-Reranker-4B", # 默认用 4B
            "rerank-0.6b": "Qwen3-Reranker-0.6B",
            "rerank-4b": "Qwen3-Reranker-4B",
            "rerank-8b": "Qwen3-Reranker-8B",
        }
        model_name = method_model_map[method]
        reranked_scores = rerank(query=query_text, docs=[item['sentence'] for item in hits_list], model_name=model_name)['scores']
        for idx, item in enumerate(hits_list):
            item['rerank_score'] = reranked_scores[idx]
        hits_list.sort(key = lambda x: -x["rerank_score"])
        
    end_time = time.time()
    return JSONResponse(
            {
                "results": hits_list, 
                "info": "version 3.0",
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
        item['hit_coarse_count'] = item_coarse_count

    filtered_results.sort(key= lambda item: -item['hit_coarse_count'])

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
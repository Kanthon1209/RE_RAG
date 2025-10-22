from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pymilvus import connections, Collection
from difflib import SequenceMatcher

import logging
import time
import json

from src.qwen.embedding import EmbeddingClient
from src.utils.misc import timer



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

def compute_entity_overlap_score(test_coarse_types: list[str], train_entities: list[dict]) -> float:
    """
    计算训练样本 entities 的 coarse_type 与测试样本 coarse_types 的 overlap score
    - 在测试 sample 内的类型加分
    - 在测试 sample 外的类型扣分
    - 返回范围 [-1, 1]
    """
    if not train_entities:
        return 0.0
    
    test_set = set(test_coarse_types)
    train_set = set(e.get("coarse_type") for e in train_entities if "coarse_type" in e)
    
    if not train_set:
        return 0.0

    n_in = len(train_set & test_set)
    n_out = len(train_set - test_set)
    
    score = (n_in - n_out) / len(train_set)  # [-1,1]
    return score

def compute_entity_name_match_score(test_sentence: str, train_entities: list[dict]) -> float:
    """
    计算训练样本中实体名与测试样本句子的匹配得分。
    - 所有实体都出现在 test_sentence 中 → 得分最高（~1）
    - 部分匹配 → 中等分
    - 全部不出现 → 负分
    """
    if not train_entities:
        return 0.0

    sentence = test_sentence.lower()
    total = len(train_entities)
    score_sum = 0.0

    for ent in train_entities:
        name = ent.get("name", "").strip().lower()
        if not name:
            continue

        if name in sentence:
            s = 1.0  # 完全匹配
        else:
            # 部分匹配判断
            ratio = SequenceMatcher(None, name, sentence).ratio()
            if ratio > 0.6:
                s = 0.5 * ratio  # 部分匹配
            else:
                s = -0.5  # 缺失惩罚

        score_sum += s

    final_score = score_sum / total
    # 归一化到 [-1, 1]
    return max(min(final_score, 1.0), -1.0)


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
                    d['overlap_score'] = overlap_score
                else:
                    d['overlap_score'] = 0
            hits_list.append(d)

    # 根据度量类型来
    hits_list.sort(key=lambda x: (-x['overlap_score'], -x['score'])) # 按照 overlap_score 先排序, 后 score 排序
    end_time = time.time()
    return JSONResponse(
            {
                "results": hits_list, 
                "help": 
                """1. Choosable return fields: ...\n2. default limit value is 10\n3. use coarse_types filed to specify the range of the recognition\n4. 训练样本中, 返回的三元组中的粗粒度标签可能不在要求的 coarse_types 中, 如果需要过滤这一部分的三元组, 避免他们出现在 output 中, 指定 filter_coarse_type 字段为 true, 注意, 不要使用引号包裹 true; filter_coarse_type 不指定值的时候, 等同于设置为 false\n5. """,
                "info": "version 1.0",
                "cost_time": f"{end_time - start_time:.4f}s"
            }, status_code=200)

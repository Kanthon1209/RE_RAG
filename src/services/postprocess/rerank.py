from difflib import SequenceMatcher
from rapidfuzz import fuzz

def compute_entity_overlap_score(test_coarse_types: list[str], train_entities: list[dict]) -> float:
    """
    计算训练样本 entities 的 coarse_type 与测试样本 coarse_types 的 overlap score
    - 完全命中 +1
    - 映射命中 +0.5
    - 不命中 0
    - 同一 coarse_type 不重复加分
    - 最终得分范围 [0, 1]
    """
    COARSE_TYPE_MAP = {
        "person": ["person", "人"],
        "人": ["person", "人"],
        "politics": ["politics", "政治"],
        "政治": ["politics", "政治"],
        "location": ["location", "位置"],
        "位置": ["location", "位置"],
        "literature": ["literature", "文学"],
        "文学": ["literature", "文学"],
        "product": ["product", "产品"],
        "产品": ["product", "产品"],
        "computer science": ["computer science", "计算机科学", "science"],
        "计算机科学": ["computer science", "计算机科学", "science"],
        "music": ["music", "音乐"],
        "音乐": ["music", "音乐"],
        "science": ["science", "科学", "biology"],
        "科学": ["science", "科学", "biology"],
        "event": ["event", "事件"],
        "事件": ["event", "事件"],
        "organisation": ["organisation", "organization", "组织", "组织机构"],
        "组织": ["organisation", "organization", "组织", "组织机构"],
        "medicine": ["medicine", "医学"],
        "医学": ["medicine", "医学"]
    }

    if not train_entities:
        return 0.0

    test_set = set(test_coarse_types)
    train_set = {e.get("coarse_type") for e in train_entities if "coarse_type" in e and e["coarse_type"]}

    score = 0.0
    matched_train_types = set()  # 防止重复加分

    for t_type in train_set:
        if t_type in matched_train_types:
            continue  # 跳过重复类型
        # 完全命中
        if t_type in test_set:
            score += 1.0
            matched_train_types.add(t_type)
        else:
            # 映射命中
            for test_type in test_set:
                mapped = COARSE_TYPE_MAP.get(test_type, [])
                if t_type in mapped:
                    score += 0.5
                    matched_train_types.add(t_type)
                    break  # 只加一次分

    # 归一化
    return min(score / len(test_set), 1.0)


def compute_entity_name_match_score(test_sentence: str, train_entities: list[dict]) -> float:
    """
    计算训练样本中实体名与测试样本句子的匹配得分。
    - 所有实体都出现在 test_sentence 中 → 得分最高（~1）
    - 部分匹配 → 中等分
    - 全部不出现 → 负分
    """
    if not train_entities:
        return 0.0

    sentence = test_sentence.lower() # 不会对中文造成影响
    total = len(train_entities)
    score_sum = 0.0

    for ent in train_entities:
        name = ent.get("name", "").strip().lower()
        if not name:
            continue
        if name in sentence:
            score_sum += 1
            # print(f'{name} in sentence')
        else:
            score = fuzz.partial_ratio(name, sentence) / 100  # 转成 [0, 1]
            # print(f'{name} score: {score}')
            score_sum += score

    final_score = score_sum# 需要平均
    # 归一化到 [-1, 1]
    return final_score
from difflib import SequenceMatcher
from rapidfuzz import fuzz

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
    
    score = (n_in - n_out) / len(test_set)  # [-1,1] # 改成了 test_set
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
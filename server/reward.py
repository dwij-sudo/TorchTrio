from typing import List, Tuple


def clamp(value: float, min_value: float = -1.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def aggregate(weighted_scores: List[Tuple[float, float]]) -> float:
    total_weight = sum(w for w, _ in weighted_scores)
    if total_weight == 0:
        return 0.0
    value = sum(w * s for w, s in weighted_scores) / total_weight
    return clamp(value)

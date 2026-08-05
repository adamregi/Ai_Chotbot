import numpy as np


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    first, second = np.array(v1), np.array(v2)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        raise ValueError("Cannot compare zero-length vectors.")
    return float(np.dot(first, second) / denominator)

from typing import Dict, Set, List

import numpy as np
from numpy.typing import NDArray


def __generate_band_buckets(band_index: int, m: NDArray, r: int, sep: str) -> Dict[str, Set[int]]:
    n, N = m.shape
    buckets: Dict[str, Set[int]] = {}

    for j in range(N):
        key = sep.join(map(str, m[(band_index * r):((band_index + 1) * r), j]))
        buckets.setdefault(key, set()).add(j)

    return buckets


def __construct_candidate_pairs(buckets_per_band: List[Dict[str, Set[int]]], m: NDArray) -> NDArray:
    n, N = m.shape
    same_bucket_counts = np.zeros((N, N))

    for band_map in buckets_per_band:
        for bucket in band_map.values():
            for i in bucket:
                for j in bucket:
                    if i == j:
                        continue

                    same_bucket_counts[min(i, j), max(i, j)] += 1

    same_bucket_counts /= 2
    candidates = np.argwhere(same_bucket_counts >= 1)

    return candidates


def lsh(m: NDArray, b: int, r: int, sep='-'):
    if len(m.shape) != 2:
        raise ValueError("Invalid m shape")

    buckets_per_band = [__generate_band_buckets(band, m, r, sep) for band in range(b)]
    return __construct_candidate_pairs(buckets_per_band, m)

from typing import List, Set

import numpy as np
from numpy.typing import NDArray


def minhash(nonzero_indices: List[Set[int]], n: int, vocab_size: int) -> NDArray:
    """
    :param nonzero_indices: a set of nonzero indices for each product
    :param n: the number of permutations to generate
    :param vocab_size: the size of shingle vocabulary
    :return: [n x m] the resulting M matrix
    """
    a = np.random.randint(low=2147483648, high=4294967296, size=n)
    b = np.random.randint(low=2147483648, high=4294967296, size=n)
    p = 864158203

    if p < vocab_size:
        raise ValueError("p < vocab_size")

    m = np.inf * np.ones((n, len(nonzero_indices)))

    h_ir = np.array([(a * r + b) % p for r in range(vocab_size)]).T

    for c, indices in enumerate(nonzero_indices):
        # indices contain all 1 indices for column c
        for r in indices:
            # for each element, take minimum of current value and hashes for row r
            m[:, c] = np.min(np.stack([m[:, c], h_ir[:, r]]), axis=0)

    return m.astype(int)

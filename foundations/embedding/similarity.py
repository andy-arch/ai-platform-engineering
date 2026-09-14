"""
similarity.py
-------------
Vector similarity and distance metrics, plus a top-k ranking helper.

All functions accept both plain Python lists and numpy arrays, and work on
1D vectors. Batch-oriented helpers operate on a query vector against a
matrix of candidate vectors.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


def _as_vector(x: ArrayLike) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape {v.shape}")
    return v


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """
    Cosine similarity: the cosine of the angle between two vectors.
    Range: [-1, 1] where 1 = identical direction, 0 = orthogonal,
    -1 = opposite direction. Ignores vector magnitude -- this is why it's
    the default metric for text embeddings, where magnitude often reflects
    text length rather than meaning.

    Returns 0.0 if either vector is all-zero (undefined angle, handled
    gracefully rather than raising a divide-by-zero error).
    """
    va, vb = _as_vector(a), _as_vector(b)
    if va.shape != vb.shape:
        raise ValueError(f"Vector shape mismatch: {va.shape} vs {vb.shape}")

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(va, vb) / (norm_a * norm_b))


def dot_product(a: ArrayLike, b: ArrayLike) -> float:
    """
    Raw dot product similarity. Sensitive to magnitude as well as direction.
    Useful when your embeddings are already normalized (then it equals
    cosine similarity) or when magnitude is meaningful for your use case.
    """
    va, vb = _as_vector(a), _as_vector(b)
    if va.shape != vb.shape:
        raise ValueError(f"Vector shape mismatch: {va.shape} vs {vb.shape}")
    return float(np.dot(va, vb))


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> float:
    """
    Straight-line (L2) distance between two vectors.
    Range: [0, inf) where 0 = identical vectors. Lower is more similar,
    which is the opposite convention from cosine similarity -- watch for
    this when ranking results.
    """
    va, vb = _as_vector(a), _as_vector(b)
    if va.shape != vb.shape:
        raise ValueError(f"Vector shape mismatch: {va.shape} vs {vb.shape}")
    return float(np.linalg.norm(va - vb))


def euclidean_to_similarity(distance: float) -> float:
    """
    Convert a euclidean distance into a bounded (0, 1] similarity score,
    so it can be compared/ranked on the same footing as cosine similarity.
    Higher = more similar. Purely a convenience transform, not a standard
    formula -- documented here so it's never mistaken for cosine similarity.
    """
    return 1.0 / (1.0 + distance)


def top_k_similar(
    query: ArrayLike,
    candidates: np.ndarray,
    k: int = 5,
    metric: str = "cosine",
) -> List[Tuple[int, float]]:
    """
    Rank a matrix of candidate vectors against a single query vector and
    return the top-k (index, score) pairs, highest similarity first.

    Args:
        query: 1D query vector, shape (d,)
        candidates: 2D array of candidate vectors, shape (n, d)
        k: number of top results to return (clamped to n if n < k)
        metric: "cosine", "dot", or "euclidean"

    Returns:
        List of (original_index, score) tuples, length min(k, n),
        sorted by score descending (for all three metrics -- euclidean
        distances are converted to a similarity score first, so "higher
        is always better" holds regardless of metric).
    """
    q = _as_vector(query)
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 2:
        raise ValueError(f"Expected candidates to be 2D, got shape {candidates.shape}")
    if candidates.shape[1] != q.shape[0]:
        raise ValueError(
            f"Dimension mismatch: query has dim {q.shape[0]}, "
            f"candidates have dim {candidates.shape[1]}"
        )

    n = candidates.shape[0]
    if n == 0:
        return []

    if metric == "cosine":
        scores = np.array([cosine_similarity(q, candidates[i]) for i in range(n)])
    elif metric == "dot":
        scores = np.array([dot_product(q, candidates[i]) for i in range(n)])
    elif metric == "euclidean":
        scores = np.array(
            [euclidean_to_similarity(euclidean_distance(q, candidates[i])) for i in range(n)]
        )
    else:
        raise ValueError(f"Unknown metric '{metric}'. Use 'cosine', 'dot', or 'euclidean'.")

    k = min(k, n)
    # argpartition for O(n) selection, then sort just the top-k slice
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    return [(int(i), float(scores[i])) for i in top_idx]

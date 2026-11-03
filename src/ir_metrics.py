"""IR evaluation metrics shared by Part B and Part C.

A ranked list is a list of document IDs, best first. `rel` is the set of relevant IDs.
"""
import math


def precision(ranked, rel):
    return sum(d in rel for d in ranked) / len(ranked) if ranked else 0.0


def recall(ranked, rel):
    return sum(d in rel for d in ranked) / len(rel) if rel else 0.0


def f1(ranked, rel):
    p, r = precision(ranked, rel), recall(ranked, rel)
    return 2 * p * r / (p + r) if p + r else 0.0


def precision_at_k(ranked, rel, k):
    """The denominator stays k even if fewer than k documents were returned."""
    return sum(d in rel for d in ranked[:k]) / k


def reciprocal_rank(ranked, rel):
    for rank, doc in enumerate(ranked, 1):
        if doc in rel:
            return 1 / rank
    return 0.0


def average_precision(ranked, rel):
    """Relevant documents that were not retrieved contribute 0 (division by |rel|)."""
    hits, total = 0, 0.0
    for rank, doc in enumerate(ranked, 1):
        if doc in rel:
            hits += 1
            total += hits / rank
    return total / len(rel) if rel else 0.0


def dcg(gains):
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))


def ndcg_at_k(ranked, grades, k):
    """`grades` maps doc -> gain; unjudged documents have gain 0.

    The ideal list is built from all judged documents, not just the retrieved ones.
    """
    actual = dcg([grades.get(d, 0) for d in ranked[:k]])
    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0

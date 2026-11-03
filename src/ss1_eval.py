"""Self Study 1: effectiveness metrics plus KLD and JSD for each Lucene query class.

All divergences use log base 2, so they are in bits and JSD lies in [0, 1].

Usage: python ss1_eval.py <data_dir> <results_dir>
"""
import collections
import json
import sys

import numpy as np
from scipy.stats import spearmanr

from ir_metrics import (
    average_precision,
    f1,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall,
    reciprocal_rank,
)
from part_c_eval import load_qrels, load_run

TYPES = [
    "Term", "Phrase", "Range-Numeric", "Range-Lexical", "Boolean",
    "Prefix", "Wildcard", "Fuzzy", "MatchAll", "OR-Baseline",
]
BASELINE = "OR-Baseline"
LAMBDA = 0.1  # weight of the collection model in term-level smoothing
EPSILON = 0.01  # weight of the uniform distribution in document-level smoothing


class Collection:
    """Term distributions for every document and for the whole collection."""

    def __init__(self, path):
        docs = {}
        for line in open(path):
            parts = line.rstrip("\n").split("\t")
            docs[parts[0]] = parts[1].split() if len(parts) > 1 else []
        vocabulary = {w: i for i, w in enumerate(sorted({w for t in docs.values() for w in t}))}
        self.ids = list(docs)
        self.position = {doc: i for i, doc in enumerate(self.ids)}
        counts = np.zeros((len(self.ids), len(vocabulary)))
        for doc, terms in docs.items():
            for word in terms:
                counts[self.position[doc], vocabulary[word]] += 1
        self.model = counts.sum(0) / counts.sum()
        self.doc_models = counts / np.maximum(counts.sum(1, keepdims=True), 1)
        self.uniform = np.ones(len(self.ids)) / len(self.ids)

    def language_model(self, doc_ids, lam=LAMBDA):
        """Mean document model mixed with the collection, so every term has P > 0."""
        rows = [self.position[d] for d in doc_ids]
        p = self.doc_models[rows].mean(0) if rows else self.model
        return (1 - lam) * p + lam * self.model

    def rank_distribution(self, ranked, eps=EPSILON):
        """Rank-discounted distribution over documents: p(d) proportional to 1/log2(1+rank)."""
        p = np.zeros(len(self.ids))
        for rank, doc in enumerate(ranked, 1):
            p[self.position[doc]] = 1 / np.log2(rank + 1)
        if p.sum() == 0:
            return self.uniform
        return (1 - eps) * p / p.sum() + eps * self.uniform


def kld(p, q):
    return float(np.sum(p * np.log2(p / q)))


def jsd(p, q):
    m = (p + q) / 2
    return 0.5 * kld(p, m) + 0.5 * kld(q, m)


def load_hit_counts(path):
    hits = collections.defaultdict(dict)
    for line in list(open(path))[1:]:
        query, query_type, count = line.split("\t")
        hits[query_type][query] = int(count)
    return hits


def effectiveness(ranked, rel, gains):
    return {
        "P10": precision_at_k(ranked, rel, 10),
        "R100": recall(ranked, rel),
        "F1": f1(ranked[:10], rel),
        "P3": precision_at_k(ranked, rel, 3),
        "AP": average_precision(ranked, rel),
        "RR": reciprocal_rank(ranked, rel),
        "nDCG10": ndcg_at_k(ranked, gains, 10),
    }


def summarise_type(query_type, runs, queries, relevant, gains, hits, coll):
    run = runs[query_type]
    per_query = [effectiveness(run.get(q, []), relevant[q], gains[q]) for q in queries]
    row = {key: mean(scores[key] for scores in per_query) for key in per_query[0]}

    kld_rel, jsd_rel, kld_coll, jsd_coll, kld_doc = [], [], [], [], []
    for query in queries:
        ranked = run.get(query, [])
        rel_model = coll.language_model(sorted(relevant[query]))
        top_model = coll.language_model(ranked[:10])
        kld_rel.append(kld(rel_model, top_model))
        jsd_rel.append(jsd(rel_model, top_model))
        kld_coll.append(kld(top_model, coll.model))
        jsd_coll.append(jsd(top_model, coll.model))
        baseline = coll.rank_distribution(runs[BASELINE].get(query, []))
        kld_doc.append(kld(baseline, coll.rank_distribution(ranked)))

    counts = [hits[query_type][q] for q in queries]
    row.update(
        KLD_rel=mean(kld_rel),
        JSD_rel=mean(jsd_rel),
        KLD_coll=mean(kld_coll),
        JSD_coll=mean(jsd_coll),
        KLD_base_doc=mean(kld_doc),
        mean_hits=mean(counts),
        median_hits=float(np.median(counts)),
        zero=sum(count == 0 for count in counts),
    )
    return row


def pairwise_jsd(runs, queries, coll):
    return {
        a: {
            b: mean(
                jsd(
                    coll.rank_distribution(runs[a].get(q, [])),
                    coll.rank_distribution(runs[b].get(q, [])),
                )
                for q in queries
            )
            for b in TYPES
        }
        for a in TYPES
    }


def kld_asymmetry(runs, queries, coll):
    """KLD in both directions against the baseline, at term level."""
    result = {}
    for query_type in TYPES:
        forward, backward = [], []
        for q in queries:
            base = coll.language_model(runs[BASELINE].get(q, [])[:10])
            other = coll.language_model(runs[query_type].get(q, [])[:10])
            forward.append(kld(base, other))
            backward.append(kld(other, base))
        result[query_type] = (mean(forward), mean(backward))
    return result


def smoothing_sensitivity(table, runs, queries, relevant, coll):
    """Do the KLD orderings hold under other smoothing weights? (Spearman against the default.)"""
    sensitivity = {}
    base_term = [table[t]["KLD_rel"] for t in TYPES]
    for lam in [0.01, 0.5]:
        values = {
            t: mean(
                kld(
                    coll.language_model(sorted(relevant[q]), lam),
                    coll.language_model(runs[t].get(q, [])[:10], lam),
                )
                for q in queries
            )
            for t in TYPES
        }
        rho = spearmanr(base_term, [values[t] for t in TYPES]).statistic
        sensitivity[f"lambda={lam}"] = {"values": values, "rho": float(rho)}

    compared = [t for t in TYPES if t != BASELINE]
    base_doc = [table[t]["KLD_base_doc"] for t in compared]
    for eps in [0.001, 0.1]:
        values = {
            t: mean(
                kld(
                    coll.rank_distribution(runs[BASELINE].get(q, []), eps),
                    coll.rank_distribution(runs[t].get(q, []), eps),
                )
                for q in queries
            )
            for t in compared
        }
        rho = spearmanr(base_doc, [values[t] for t in compared]).statistic
        sensitivity[f"eps={eps}"] = {"values": values, "rho": float(rho)}
    return sensitivity


def main(data, results_dir):
    relevant, gains = load_qrels(f"{data}/qrels.txt")
    queries = sorted(relevant, key=int)
    runs = {t: load_run(f"{results_dir}/ss1_{t}.run") for t in TYPES}
    hits = load_hit_counts(f"{results_dir}/ss1_hit_counts.tsv")
    coll = Collection(f"{results_dir}/docs_analyzed.tsv")

    table = {t: summarise_type(t, runs, queries, relevant, gains, hits, coll) for t in TYPES}
    correlation = spearmanr([table[t]["JSD_rel"] for t in TYPES], [table[t]["AP"] for t in TYPES])
    overlap = {
        t: mean(
            len(set(runs[t].get(q, [])[:10]) & set(runs[BASELINE].get(q, [])[:10])) / 10
            for q in queries
        )
        for t in TYPES
    }
    output = {
        "tab": table,
        "J": pairwise_jsd(runs, queries, coll),
        "asym": kld_asymmetry(runs, queries, coll),
        "overlap": overlap,
        "corr": {"rho": float(correlation.statistic), "p": float(correlation.pvalue)},
        "V": len(coll.model),
        "lam": LAMBDA,
        "eps": EPSILON,
        "sens": smoothing_sensitivity(table, runs, queries, relevant, coll),
    }
    json.dump(output, open(f"{results_dir}/ss1_results.json", "w"), indent=1)

    columns = ["mean_hits", "P3", "P10", "R100", "AP", "RR", "nDCG10", "KLD_rel", "JSD_rel"]
    print(f"{'type':14}" + "".join(f"{c:>10}" for c in columns))
    for t in TYPES:
        print(f"{t:14}" + "".join(f"{table[t][c]:10.3f}" for c in columns))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

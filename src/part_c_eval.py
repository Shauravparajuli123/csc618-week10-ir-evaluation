"""Part C: evaluate saved runs with the Part B metric functions and test the differences.

Usage: python part_c_eval.py <data_dir> <results_dir>
"""
import collections
import json
import sys

import numpy as np
from scipy.stats import wilcoxon

from ir_metrics import (
    average_precision,
    f1,
    mean,
    ndcg_at_k,
    precision,
    precision_at_k,
    recall,
    reciprocal_rank,
)

MODELS = ["TF-IDF", "BM25", "LM-Dirichlet"]
VARIANTS = [
    "LMD-mu50", "LMD-mu100", "LMD-mu200", "LMD-mu500", "LMD-mu1000",
    "LMJM-l0.1", "LMJM-l0.3", "LMJM-l0.7",
    "BM25-b0.25", "BM25-b0.5", "BM25-b1.0",
]
PAIRS = [("BM25", "TF-IDF"), ("BM25", "LM-Dirichlet"), ("TF-IDF", "LM-Dirichlet")]
TESTED_METRICS = ["AP", "nDCG10", "RR", "P10"]
PERMUTATIONS = 100_000
SEED = 618


def load_qrels(path):
    """Returns relevant sets and gain dictionaries. This qrel copy is binary, so gain = 1."""
    relevant, gains = collections.defaultdict(set), collections.defaultdict(dict)
    for line in open(path):
        query, _, doc, label = line.split()
        if int(label) > 0:
            relevant[query].add(doc)
            gains[query][doc] = 1
    return relevant, gains


def load_run(path):
    """Reads a TREC-format run file into {query: [doc, doc, ...]} in rank order."""
    run = collections.defaultdict(list)
    for line in open(path):
        parts = line.split()
        run[parts[0]].append(parts[2])
    return run


def evaluate(run, relevant, gains):
    results = {}
    for query, rel in relevant.items():
        ranked = run.get(query, [])
        results[query] = {
            "P": precision(ranked, rel),
            "R": recall(ranked, rel),
            "F1": f1(ranked, rel),
            "P10": precision_at_k(ranked, rel, 10),
            "R10": recall(ranked[:10], rel),
            "F1_10": f1(ranked[:10], rel),
            "P3": precision_at_k(ranked, rel, 3),
            "AP": average_precision(ranked, rel),
            "RR": reciprocal_rank(ranked, rel),
            "nDCG5": ndcg_at_k(ranked, gains[query], 5),
            "nDCG10": ndcg_at_k(ranked, gains[query], 10),
        }
    return results


def average(per_query):
    keys = next(iter(per_query.values()))
    return {key: mean(scores[key] for scores in per_query.values()) for key in keys}


def paired_tests(per_model, queries):
    """Two-sided sign-flip randomization test and Wilcoxon signed-rank test on each pair."""
    rng = np.random.default_rng(SEED)
    tests = []
    for a, b in PAIRS:
        for metric in TESTED_METRICS:
            x = np.array([per_model[a][q][metric] for q in queries])
            y = np.array([per_model[b][q][metric] for q in queries])
            diff = x - y
            signs = rng.choice([-1, 1], size=(PERMUTATIONS, len(diff)))
            p_perm = (np.abs((signs * diff).mean(axis=1)) >= abs(diff.mean()) - 1e-12).mean()
            p_wilcoxon = wilcoxon(x, y, zero_method="wilcox").pvalue if np.any(diff) else 1.0
            tests.append(
                {
                    "a": a,
                    "b": b,
                    "metric": metric,
                    "diff": float(diff.mean()),
                    "wins": int((diff > 0).sum()),
                    "losses": int((diff < 0).sum()),
                    "ties": int((diff == 0).sum()),
                    "p_perm": float(p_perm),
                    "p_wilcoxon": float(p_wilcoxon),
                }
            )
    return tests


def best_model_counts(per_model, queries):
    counts = collections.Counter()
    for query in queries:
        scores = {m: per_model[m][query]["AP"] for m in MODELS}
        top = max(scores.values())
        winners = [m for m in MODELS if abs(scores[m] - top) < 1e-12]
        counts["tie" if len(winners) > 1 else winners[0]] += 1
    return dict(counts)


def largest_spread_examples(per_model, queries, relevant, texts, n=5):
    def spread(query):
        scores = [per_model[m][query]["AP"] for m in MODELS]
        return -(max(scores) - min(scores))

    return [
        {
            "q": query,
            "text": texts[query],
            "nrel": len(relevant[query]),
            **{m: per_model[m][query]["AP"] for m in MODELS},
        }
        for query in sorted(queries, key=spread)[:n]
    ]


def effect_of_raw_query_ids(data, results_dir, relevant, gains):
    """Shows what happens if the non-consecutive query numbers are not renumbered."""
    raw_number = {
        line.split("\t")[0]: line.split("\t")[1] for line in open(f"{data}/queries.tsv")
    }
    run = load_run(f"{results_dir}/partc_BM25.run")
    wrong = evaluate({raw_number[q]: docs for q, docs in run.items()}, relevant, gains)
    return {key: mean(scores[key] for scores in wrong.values()) for key in ["AP", "P10", "RR"]}


def write_per_query_csv(path, per_model, queries):
    keys = list(next(iter(per_model["BM25"].values())))
    with open(path, "w") as f:
        f.write("model,query," + ",".join(keys) + "\n")
        for model in MODELS:
            for query in queries:
                values = ",".join(f"{per_model[model][query][k]:.6f}" for k in keys)
                f.write(f"{model},{query},{values}\n")


def main(data, results_dir):
    relevant, gains = load_qrels(f"{data}/qrels.txt")
    queries = sorted(relevant, key=int)
    texts = {line.split("\t")[0]: line.split("\t")[2].strip() for line in open(f"{data}/queries.tsv")}

    runs = {m: load_run(f"{results_dir}/partc_{m}.run") for m in MODELS + VARIANTS}
    per_model = {m: evaluate(runs[m], relevant, gains) for m in MODELS + VARIANTS}

    output = {
        "agg": {m: average(per_model[m]) for m in MODELS + VARIANTS},
        "wrong_ids": effect_of_raw_query_ids(data, results_dir, relevant, gains),
        "tests": paired_tests(per_model, sorted(relevant)),
        "best_AP": best_model_counts(per_model, queries),
        "zero_P10": {m: sum(per_model[m][q]["P10"] == 0 for q in queries) for m in MODELS},
        "examples": largest_spread_examples(per_model, queries, relevant, texts),
        "overlap10": {
            f"{a}|{b}": mean(len(set(runs[a][q][:10]) & set(runs[b][q][:10])) / 10 for q in queries)
            for a, b in PAIRS
        },
        "retrieved_mean": {m: mean(len(runs[m][q]) for q in queries) for m in MODELS},
    }
    json.dump(output, open(f"{results_dir}/partc_results.json", "w"), indent=1)
    write_per_query_csv(f"{results_dir}/partc_per_query.csv", per_model, queries)

    for model in MODELS:
        scores = output["agg"][model]
        print(f"{model:13} MAP={scores['AP']:.4f} MRR={scores['RR']:.4f} nDCG@10={scores['nDCG10']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

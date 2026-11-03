"""Cross-check our Part C metrics against the independent ranx library.

Usage: python verify.py <data_dir> <results_dir>
"""
import json
import sys

from ranx import Qrels, Run, evaluate

from part_c_eval import MODELS, load_qrels

METRIC_PAIRS = {
    "map": ("AP", "map"),
    "mrr": ("RR", "mrr"),
    "p3": ("P3", "precision@3"),
    "p10": ("P10", "precision@10"),
    "r10": ("R10", "recall@10"),
    "n10": ("nDCG10", "ndcg@10"),
    "n5": ("nDCG5", "ndcg@5"),
    "r": ("R", "recall@1000"),
}


def read_run(path):
    run = {}
    for line in open(path):
        query, _, doc, rank, _score, _tag = line.split()
        run.setdefault(query, {})[doc] = 1000.0 - int(rank)
    return Run(run)


def main(data, results_dir):
    relevant, _ = load_qrels(f"{data}/qrels.txt")
    qrels = Qrels({q: {d: 1 for d in docs} for q, docs in relevant.items()})
    own = json.load(open(f"{results_dir}/partc_results.json"))["agg"]

    comparison = {}
    for model in MODELS:
        theirs = evaluate(qrels, read_run(f"{results_dir}/partc_{model}.run"),
                          [name for _, name in METRIC_PAIRS.values()])
        comparison[model] = {
            label: (round(own[model][key], 6), round(float(theirs[name]), 6))
            for label, (key, name) in METRIC_PAIRS.items()
        }
        match = all(a == b for a, b in comparison[model].values())
        print(f"{model:13} all metrics match ranx: {match}")
    json.dump(comparison, open(f"{results_dir}/verify_ranx.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

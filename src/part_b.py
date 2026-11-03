"""Part B: controlled comparison of System A and System B.

Usage: python part_b.py <output_dir>
"""
import csv
import json
import sys

from ir_metrics import (
    average_precision,
    dcg,
    f1,
    mean,
    ndcg_at_k,
    precision,
    precision_at_k,
    recall,
    reciprocal_rank,
)

DOCUMENTS = {
    "D1": "Neural networks and deep learning methods for image classification",
    "D2": "BM25 ranking functions for information retrieval",
    "D3": "Convolutional neural networks for image recognition",
    "D4": "Language models and probabilistic document retrieval",
    "D5": "Image classification using machine learning",
    "D6": "BM25, query likelihood, and document ranking",
    "D7": "Unigram language models for text search",
    "D8": "Web crawling and link analysis",
}
QUERIES = {
    "Q1": "Neural networks for image classification",
    "Q2": "BM25 information retrieval",
    "Q3": "Language models and document retrieval",
}
RELEVANT = {"Q1": {"D1", "D3", "D5"}, "Q2": {"D2", "D4", "D6"}, "Q3": {"D4", "D6", "D7"}}
GRADES_Q1 = {"D1": 3, "D3": 2, "D5": 1}
RUNS = {
    "A": {
        "Q1": ["D1", "D2", "D3", "D5", "D4"],
        "Q2": ["D2", "D1", "D4", "D6", "D3"],
        "Q3": ["D4", "D2", "D6", "D7", "D1"],
    },
    "B": {
        "Q1": ["D2", "D1", "D4", "D3", "D6"],
        "Q2": ["D1", "D4", "D2", "D5", "D6"],
        "Q3": ["D1", "D4", "D3", "D7", "D6"],
    },
}
METRICS = ["Precision", "Recall", "F1", "P3", "AP", "RR"]


def per_query_rows():
    rows = []
    for system, run in RUNS.items():
        for query, ranked in run.items():
            rel = RELEVANT[query]
            rows.append(
                {
                    "System": system,
                    "Query": query,
                    "Precision": precision(ranked, rel),
                    "Recall": recall(ranked, rel),
                    "F1": f1(ranked, rel),
                    "P3": precision_at_k(ranked, rel, 3),
                    "AP": average_precision(ranked, rel),
                    "RR": reciprocal_rank(ranked, rel),
                }
            )
    return rows


def aggregate(rows):
    results = {}
    for system, run in RUNS.items():
        selected = [row for row in rows if row["System"] == system]
        gains = [GRADES_Q1.get(doc, 0) for doc in run["Q1"][:5]]
        results[system] = {
            "MAP": mean(row["AP"] for row in selected),
            "MRR": mean(row["RR"] for row in selected),
            "mean_P": mean(row["Precision"] for row in selected),
            "mean_R": mean(row["Recall"] for row in selected),
            "mean_F1": mean(row["F1"] for row in selected),
            "mean_P3": mean(row["P3"] for row in selected),
            "gains_Q1": gains,
            "DCG5_Q1": dcg(gains),
            "IDCG5_Q1": dcg(sorted(GRADES_Q1.values(), reverse=True)),
            "nDCG5_Q1": ndcg_at_k(run["Q1"], GRADES_Q1, 5),
        }
    return results


def main(out):
    rows = per_query_rows()
    results = aggregate(rows)

    with open(f"{out}/partb_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json.dump({"rows": rows, "agg": results}, open(f"{out}/partb_results.json", "w"), indent=1)

    print(f"{'System':6} {'Query':5} " + " ".join(f"{m:>9}" for m in METRICS))
    for row in rows:
        values = " ".join(f"{row[m]:9.4f}" for m in METRICS)
        print(f"{row['System']:6} {row['Query']:5} {values}")
    for system, values in results.items():
        print(
            f"System {system}: MAP={values['MAP']:.4f} MRR={values['MRR']:.4f} "
            f"nDCG@5(Q1)={values['nDCG5_Q1']:.4f}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")

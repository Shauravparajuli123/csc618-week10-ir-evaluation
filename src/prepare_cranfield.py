"""Convert the TREC-XML copy of Cranfield to flat files and validate it.

Query IDs are renumbered 1..225 by file position: the raw <num> values are
non-consecutive (1, 2, 4, 8, ...) while the qrels number queries consecutively.

Usage: python prepare_cranfield.py <cranfield_repo_dir> <output_dir>
"""
import collections
import json
import re
import sys


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def field(record, tag):
    match = re.search(fr"<{tag}>(.*?)</{tag}>", record, flags=re.S)
    return clean(match[1]) if match else ""


def read_docs(path):
    raw = open(path, encoding="utf-8").read()
    records = re.findall(r"<doc>(.*?)</doc>", raw, flags=re.S)
    return [(field(r, "docno"), field(r, "title"), field(r, "text")) for r in records]


def read_queries(path):
    raw = open(path, encoding="utf-8").read()
    pattern = r"<num>\s*(\d+)\s*</num>\s*<title>(.*?)</title>"
    return [(int(number), clean(text)) for number, text in re.findall(pattern, raw, flags=re.S)]


def main(root, out):
    docs = read_docs(f"{root}/cran.all.1400.xml")
    queries = read_queries(f"{root}/cran.qry.xml")
    qrels = [line.split() for line in open(f"{root}/cranqrel.trec.txt")]

    with open(f"{out}/docs.tsv", "w") as f:
        for doc_id, title, text in docs:
            f.write(f"{doc_id}\t{title}\t{text}\n")
    with open(f"{out}/queries.tsv", "w") as f:
        for sequential_id, (number, text) in enumerate(queries, 1):
            f.write(f"{sequential_id}\t{number}\t{text}\n")
    with open(f"{out}/qrels.txt", "w") as f:
        for query, _, doc, relevance in qrels:
            f.write(f"{query} 0 {doc} {relevance}\n")

    doc_ids = {d[0] for d in docs}
    relevant = collections.Counter(q for q, _, d, r in qrels if int(r) > 0)
    stats = {
        "docs": len(docs),
        "unique_doc_ids": len(doc_ids),
        "empty_text_docs": [d[0] for d in docs if not d[2]],
        "title_is_prefix_of_text": sum(d[2].startswith(d[1][:40]) for d in docs),
        "queries": len(queries),
        "raw_max_query_num": max(number for number, _ in queries),
        "qrel_rows": len(qrels),
        "qrel_query_ids": len({r[0] for r in qrels}),
        "qrel_max_query_id": max(int(r[0]) for r in qrels),
        "qrel_label_counts": dict(collections.Counter(r[3] for r in qrels)),
        "qrel_docs_missing": sum(r[2] not in doc_ids for r in qrels),
        "queries_with_relevant": len(relevant),
        "relevant_per_query_mean": sum(relevant.values()) / max(1, len(relevant)),
        "relevant_per_query_min": min(relevant.values()),
        "relevant_per_query_max": max(relevant.values()),
    }
    json.dump(stats, open(f"{out}/dataset_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

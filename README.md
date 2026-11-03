# CSC 618 Week 10: Information Retrieval Evaluation

Coursework for CSC 618 (Information Retrieval), Utica University, Fall 2026. Author: Suman KC.

The project compares Lucene query types, evaluates two hypothetical systems with standard IR metrics, and compares three retrieval models on the Cranfield test collection.

| Task | What it does | Main code |
|---|---|---|
| Self Study 1 | Runs 8 Lucene query types; compares them using metrics, KLD and JSD | `src/LuceneExperiment.java`, `src/ss1_eval.py` |
| Part B | Evaluates System A and System B on a small fixed dataset | `src/part_b.py`, `src/ir_metrics.py` |
| Part C | Runs TF-IDF, BM25 and a Language Model on Cranfield | `src/LuceneExperiment.java`, `src/part_c_eval.py` |

## Results

**Part B**

| System | MAP | MRR | nDCG@5 (Q1) |
|---|---|---|---|
| A | 0.8056 | 1.0000 | 0.9305 |
| B | 0.4852 | 0.5000 | 0.5784 |

**Part C** (225 queries, default parameters; nDCG uses binary relevance)

| Model | MAP | MRR | P@10 | nDCG@10 |
|---|---|---|---|---|
| TF-IDF | 0.2931 | 0.5327 | 0.2244 | 0.3690 |
| BM25 | 0.2952 | 0.5204 | 0.2284 | 0.3741 |
| LM-Dirichlet (mu=2000) | 0.2341 | 0.4557 | 0.1840 | 0.3041 |

BM25 and TF-IDF are statistically tied on MAP (p = 0.75). Both are significantly better than LM-Dirichlet.

## Repository layout

```
src/
  prepare_cranfield.py   Convert and validate the dataset
  LuceneExperiment.java  Build the index and run all searches
  ir_metrics.py          Metric functions used by every experiment
  part_b.py              Part B experiment
  part_c_eval.py         Part C evaluation and significance tests
  ss1_eval.py            Self Study 1 evaluation, KLD and JSD
  verify.py              Cross-check against the ranx library
scripts/get_lucene.py    Download the Lucene jar
results/                 Saved rankings and metric output
reports/                 PDF submissions
run_all.sh               Run everything
```

## Run it

Requires Java 21+, Python 3.10+ and Git. Commands are for macOS or Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

The script downloads the dataset and Lucene, which are not stored in this repository. Results are deterministic, and the randomization test uses a fixed seed.

Tested with Lucene 10.5.0, OpenJDK 21, Python 3.13, numpy 2.5.3, scipy 1.18.1 and ranx 0.3.21.

## Data and tools

- Cranfield collection: [TREC-format copy](https://github.com/oussbenk/cranfield-trec-dataset), originally from the [University of Glasgow](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/). The copy has binary relevance labels.
- [Apache Lucene](https://lucene.apache.org/) 10.5.0, obtained through the jar bundled with [Pyserini](https://github.com/castorini/pyserini) 2.4.0.
- Query IDs are renumbered 1 to 225 by file position so that they match the relevance judgments.

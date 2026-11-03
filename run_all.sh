#!/usr/bin/env bash
# Reproduces every experiment from scratch. Run from the repository root.
set -euo pipefail

DATASET_COMMIT=1208e6edfb6cb2527b2c44398d3d8fefd3249144
mkdir -p data results bin

echo "[1/7] Download Cranfield"
if [ ! -d data/cranfield-trec-dataset ]; then
  git clone -q https://github.com/oussbenk/cranfield-trec-dataset.git data/cranfield-trec-dataset
  git -C data/cranfield-trec-dataset checkout -q "$DATASET_COMMIT"
fi

echo "[2/7] Prepare data"
python3 src/prepare_cranfield.py data/cranfield-trec-dataset data > /dev/null

echo "[3/7] Get Lucene"
python3 scripts/get_lucene.py

echo "[4/7] Index and search with Lucene"
javac -nowarn -cp "lib/*" -d bin src/LuceneExperiment.java
java -cp "lib/*:bin" LuceneExperiment data results 2> /dev/null

echo "[5/7] Part B"
python3 src/part_b.py results

echo "[6/7] Part C and Self Study 1"
python3 src/part_c_eval.py data results
python3 src/ss1_eval.py data results

echo "[7/7] Cross-check with ranx"
python3 src/verify.py data results

echo "Done. Output is in results/"

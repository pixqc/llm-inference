#!/bin/bash
set -e
export HUGGINGFACE_TOKEN=""
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN must not be empty."
    exit 1
fi

pip install --upgrade torch
pip install -U \
    tiktoken \
    "huggingface_hub[cli]" \
    transformers \
    blobfile

PYTHONPATH='.' MODEL_ID=meta-llama/Llama-3.2-1B OUT_DIR=src/model/1B python3 src/download_model.py
PYTHONPATH='.' MODEL_ID=meta-llama/Llama-3.2-1B-Instruct OUT_DIR=src/model/1B-Instruct python3 src/download_model.py

mkdir -p evals/mmlu
curl -L https://people.eecs.berkeley.edu/~hendrycks/data.tar -o /tmp/data.tar
tar xf /tmp/data.tar -C /tmp
mv /tmp/data/* evals/mmlu/
rm -rf /tmp/data

mkdir -p evals/math
curl -L https://people.eecs.berkeley.edu/~hendrycks/MATH.tar -o /tmp/MATH.tar
tar xf /tmp/MATH.tar -C /tmp
mv /tmp/MATH/* evals/math/
rm -rf /tmp/MATH

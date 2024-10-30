#!/bin/bash

# once ssh'd into box
# git clone https://github.com/pixqc/llm-inference.git && mv llm-inference/* llm-inference/.* . 2>/dev/null && rmdir llm-inference

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

mkdir -p evals/mmlu evals/math
curl -L https://people.eecs.berkeley.edu/~hendrycks/data.tar | tar x --no-same-owner -C evals/mmlu
curl -L https://people.eecs.berkeley.edu/~hendrycks/MATH.tar | tar x --no-same-owner -C evals/math

PYTHONPATH='.' python3 src/torch_main.py  # test the inference

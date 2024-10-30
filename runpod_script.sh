#!/bin/bash
set -e
pip install --upgrade torch
pip install -U \
    tiktoken \
    "jax[cuda12]" \
    "huggingface_hub[cli]" \
    transformers \
    blobfile

HUGGING_FACE_TOKEN=""
huggingface-cli login --token $HUGGING_FACE_TOKEN

PYTHONPATH='.' MODEL_ID=meta-llama/Llama-3.2-1B OUT_DIR=src/model/1B python3 src/model/download_model.py
PYTHONPATH='.' MODEL_ID=meta-llama/Llama-3.2-1B-Instruct OUT_DIR=src/model/1B-Instruct python3 src/model/download_model.py
PYTHONPATH='.' python3 evals/download_evals.py

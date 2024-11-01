import json
import re
from pathlib import Path

import torch
from math_utils import MATH, load_math
from mmlu_utils import MMLU, load_mmlu
from tqdm import tqdm

from src.torch_main import LLAMA_1B_PARAMS, Llama, device


def mmlu_run(llama: Llama, mmlu_items: list[MMLU]):
  SYSTEM_PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>"
  )
  correct = total = 0
  encode_kwargs = {"bos": False, "eos": False, "allowed_special": "all"}

  # TODO: VERBOSE=1
  for mmlu in tqdm(mmlu_items[:50], desc="Evaluating MMLU"):
    total += 1
    prompt = SYSTEM_PROMPT + mmlu.format_instruct_q()
    tokens = llama.tokenizer.encode(prompt, **encode_kwargs)
    tokens = torch.tensor(tokens, device=device).reshape(1, -1)
    attn_mask = llama._build_attn_mask(tokens.shape[-1], None)
    for chunk in llama.generate(tokens, attn_mask, sampler="topk_greedy", temp=1, k=4):
      pred = chunk["choices"][0]["delta"]["content"].strip()
      answer = chr(mmlu.answer + ord("A"))
      result = {
        "dataset": "mmlu",
        "prompt": prompt,
        "pred": pred,
        "answer": answer,
        "logprobs": [
          {"token": tp["token"], "logprob": tp["logprob"]}
          for tp in chunk["top_predictions"]
        ],
      }
      print(json.dumps(result))
      correct += pred == answer
      break  # the first char is the answer

  accuracy = correct / total
  print(json.dumps({"dataset": "mmlu", "accuracy": accuracy}))


def math_run(llama: Llama, math_items: list[MATH]):
  SYSTEM_PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>"
  )
  correct = total = 0
  encode_kwargs = {"bos": False, "eos": False, "allowed_special": "all"}

  for math in tqdm(math_items[:25], desc="Evaluating MATH"):
    total += 1
    prompt = SYSTEM_PROMPT + math.format_instruct_q()
    tokens = llama.tokenizer.encode(prompt, **encode_kwargs)
    tokens = torch.tensor(tokens, device=device).reshape(1, -1)
    attn_mask = llama._build_attn_mask(tokens.shape[-1], None)
    response = ""
    for chunk in llama.generate(tokens, attn_mask, sampler="topk_greedy", temp=1, k=25):
      pred = chunk["choices"][0]["delta"]["content"]
      response += pred
      result = {
        "dataset": "math",
        "prompt": prompt,
        "pred": pred,
        "current_token_position": chunk["current_token_position"],
        "logprobs": [
          {"token": tp["token"], "logprob": tp["logprob"]}
          for tp in chunk["top_predictions"]
        ],
      }
      print(json.dumps(result))

    pred_match = re.search(r"\\boxed{([^}]*)}", response)
    true_match = re.search(r"\\boxed{([^}]*)}", math.solution)
    result = {
      "dataset": "math",
      "prompt": prompt,
      "pred_match": pred_match.group(1) if pred_match else None,
      "true_match": true_match.group(1) if true_match else None,
    }
    print(json.dumps(result))

    if pred_match and true_match and pred_match.group(1) == true_match.group(1):
      # TODO: normalize?
      correct += 1

  print(json.dumps({"dataset": "math", "accuracy": correct / total}))


if __name__ == "__main__":
  is_instruct = True
  weight_path = f"src/model/1B{'-Instruct' if is_instruct else ''}"
  llama = Llama(is_instruct, LLAMA_1B_PARAMS, weight_path, "src/tokenizer.model")

  mmlu_items = list(load_mmlu(Path("evals/mmlu/test")))
  mmlu_run(llama, mmlu_items)
  math_items = list(load_math(Path("evals/math/test")))
  math_run(llama, math_items)

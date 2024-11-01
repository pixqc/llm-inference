import json
from pathlib import Path
from typing import Iterator, List, NamedTuple

import torch

from src.torch_main import LLAMA_1B_PARAMS, Llama, device


class MATH(NamedTuple):
  problem: str
  level: str
  typ: str
  solution: str

  def prompt_template(self) -> str:
    return (
      "Solve the following math problem efficiently and clearly:\n\n"
      "- For simple problems (2 steps or fewer):\n"
      "Provide a concise solution with minimal explanation.\n\n"
      "- For complex problems (3 steps or more):\n"
      "Use this step-by-step format:\n\n"
      "## Step 1: [Concise description]\n"
      "[Brief explanation and calculations]\n\n"
      "## Step 2: [Concise description]\n"
      "[Brief explanation and calculations]\n\n"
      "...\n\n"
      "Regardless of the approach, always conclude with:\n\n"
      "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
      "Where [answer] is just the final number or expression that solves the problem.\n\n"
      f"Problem: {self.problem}"
    )

  def format_instruct_qa(self) -> str:
    prompt = self.prompt_template()
    return (
      f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
      f"<|start_header_id|>assistant<|end_header_id|>\n\n"
      f"{self.solution}<|eot_id|>"
    )

  def format_instruct_q(self) -> str:
    prompt = self.prompt_template()
    return (
      f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
      f"<|start_header_id|>assistant<|end_header_id|>\n\n"
      f"The solution is"
    )


def format_fewshot(
  examples: List[MATH],
  question: MATH,
) -> str:
  prompt = []
  prompt.extend(ex.format_instruct_qa() for ex in examples)
  prompt.append(question.format_instruct_q())
  return "".join(prompt)


def load_math(directory: Path) -> Iterator[MATH]:
  for subdir in directory.iterdir():
    if subdir.is_dir():
      for json_file in subdir.glob("*.json"):
        with open(json_file) as f:
          data = json.load(f)
          yield MATH(
            problem=data["problem"],
            level=data["level"],
            typ=data["type"],
            solution=data["solution"],
          )


if __name__ == "__main__":
  SYSTEM_PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>"
  )
  math_problems = list(load_math(Path("evals/math/test")))
  is_instruct = True
  weight_path = f"src/model/1B{'-Instruct' if is_instruct else ''}"
  llama = Llama(is_instruct, LLAMA_1B_PARAMS, weight_path, "src/tokenizer.model")

  correct = total = 0
  encode_kwargs = {"bos": False, "eos": False, "allowed_special": "all"}

  with open("evals/eval_math.jsonl", "a") as f:
    for math_item in math_problems:
      total += 1
      prompt = SYSTEM_PROMPT + math_item.format_instruct_q()
      tokens = llama.tokenizer.encode(prompt, **encode_kwargs)
      tokens = torch.tensor(tokens, device=device).reshape(1, -1)
      attn_mask = llama._build_attn_mask(tokens.shape[-1], None)
      print(llama.tokenizer.decode(tokens[0].tolist()), end="")
      for chunk in llama.generate(
        tokens, attn_mask, sampler="topk_greedy", temp=0, k=25
      ):
        pred = chunk["choices"][0]["delta"]["content"]
        print(pred, end="", flush=True)

  print(f"\nMath Accuracy: {correct}/{total} = {correct/total:.2%}")

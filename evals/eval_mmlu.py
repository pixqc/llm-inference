import csv
import json
from pathlib import Path
from typing import List, NamedTuple

from src.mlx_main import LLAMA_1B_PARAMS, Llama


class MMLU(NamedTuple):
  question: str
  choices: List[str]
  answer: int


def read_mmlu(directory: Path) -> List[MMLU]:
  return [
    MMLU(row[0], row[1:5], ord(row[5]) - ord("A"))
    for csv_file in directory.glob("*.csv")
    for row in csv.reader(open(csv_file))
    if len(row) == 6
  ]


def format_q(mmlu: MMLU) -> str:
  return (
    "Given the following question and four candidate answers (A, B, C and D), choose the best answer."
    f"\nQuestion: {mmlu.question}\n"
    + "\n".join(f"{c}. {a}" for c, a in zip("ABCD", mmlu.choices))
    + '\nYour response should end with "The best answer is [the_answer_letter]" where the [the_answer_letter] is one of A, B, C or D.'
  )


def format_qa(mmlu: MMLU) -> str:
  return format_q(mmlu) + f" {chr(mmlu.answer + ord('A'))}.<|eot_id|>"


if __name__ == "__main__":
  MMLU_DIR = Path("evals/mmlu/test")
  mmlu = read_mmlu(MMLU_DIR)
  mmlu_zero = [format_q(data) for data in mmlu]

  is_instruct = True
  weight_path, tok_path = "src/model/1B", "src/tokenizer.model"
  weight_path = weight_path + "-Instruct" if is_instruct else weight_path
  llama = Llama(is_instruct, LLAMA_1B_PARAMS, weight_path, tok_path)

  with open("evals/eval_mmlu.jsonl", "a") as f:
    for question in mmlu_zero[:10]:
      tokens, attn_mask = llama.tokenize(
        [question],
        format_instruct=True,
        postfix="The best answer is",
      )
      print(llama.detokenize(tokens[0]), end="")
      for chunk in llama.generate(
        tokens, attn_mask, sampler="topk_greedy", temp=0, k=5
      ):
        print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
        f.write(json.dumps(chunk) + "\n")

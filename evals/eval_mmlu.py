import csv
import json
from pathlib import Path
from typing import List, NamedTuple

from src.torch_main import LLAMA_1B_PARAMS, Llama


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


def format_qa(mmlu: MMLU) -> tuple[str, str]:
  return (format_q(mmlu), chr(mmlu.answer + ord("A")))


if __name__ == "__main__":
  MMLU_DIR = Path("evals/mmlu/test")
  mmlu = read_mmlu(MMLU_DIR)
  mmlu_zero = [format_qa(data) for data in mmlu]

  is_instruct = True
  weight_path, tok_path = "src/model/1B", "src/tokenizer.model"
  weight_path = weight_path + "-Instruct" if is_instruct else weight_path
  llama = Llama(is_instruct, LLAMA_1B_PARAMS, weight_path, tok_path)

  correct, total = 0, 0
  with open("evals/eval_mmlu.jsonl", "a") as f:
    for question, answer in mmlu_zero[:10]:
      total += 1
      tokens, attn_mask = llama.tokenize(
        [question],
        format_instruct=True,
        postfix="The best answer is",
      )
      print(llama.detokenize(tokens[0]), end="")
      for chunk in llama.generate(
        tokens, attn_mask, sampler="topk_greedy", temp=0, k=4
      ):
        print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
        logprobs = chunk["top_predictions"]
        pred = chunk["choices"][0]["delta"]["content"].strip()
        logprobs = [{"token": tp["token"], "logprob": tp["logprob"]} for tp in logprobs]
        result = {
          "question": question,
          "pred": pred,
          "ground_truth": answer,
          "logprobs": logprobs,
        }
        if pred == answer:
          correct += 1
        f.write(json.dumps(result) + "\n")
        f.flush()
        break

  accuracy = correct / total
  print(f"\nMMLU Accuracy: {accuracy:.2%}")

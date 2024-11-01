import csv
from pathlib import Path
from typing import Iterator, List, NamedTuple

import torch

from src.torch_main import LLAMA_1B_PARAMS, Llama, device


class MMLU(NamedTuple):
  question: str
  choices: List[str]
  answer: int

  def prompt_template(self) -> str:
    return (
      "Given the following question and four candidate answers (A, B, C and D), choose the best answer.\n"
      f"Question: {self.question}\n"
      + "\n".join(f"{c}. {a}" for c, a in zip("ABCD", self.choices))
      + '\nYour response should end with "The best answer is [the_answer_letter]" where the [the_answer_letter] is one of A, B, C or D.'
    )

  def format_instruct_qa(self) -> str:
    prompt = self.prompt_template()
    answer = chr(self.answer + ord("A"))
    return (
      f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
      f"<|start_header_id|>assistant<|end_header_id|>\n\n"
      f"The best answer is {answer}.<|eot_id|>"
    )

  def format_instruct_q(self) -> str:
    prompt = self.prompt_template()
    return (
      f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
      f"<|start_header_id|>assistant<|end_header_id|>\n\n"
      f"The best answer is"
    )


def format_fewshot(
  examples: List[MMLU],
  question: MMLU,
) -> str:
  prompt = []
  prompt.extend(ex.format_instruct_qa() for ex in examples)
  prompt.append(question.format_instruct_q())
  return "".join(prompt)


def load_mmlu(path: Path) -> Iterator[MMLU]:
  return (
    MMLU(row[0], row[1:5], ord(row[5]) - ord("A"))
    for csv_file in path.glob("*.csv")
    for row in csv.reader(open(csv_file))
    if len(row) == 6
  )


if __name__ == "__main__":
  SYSTEM_PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>"
  )
  n_fewshot = 5
  fewshot = list(load_mmlu(Path("evals/mmlu/val")))[:n_fewshot]
  mmlu = list(load_mmlu(Path("evals/mmlu/test")))
  is_instruct = True
  weight_path = f"src/model/1B{'-Instruct' if is_instruct else ''}"
  llama = Llama(is_instruct, LLAMA_1B_PARAMS, weight_path, "src/tokenizer.model")
  correct = total = 0
  encode_kwargs = {"bos": False, "eos": False, "allowed_special": "all"}
  for mmlu_item in mmlu:
    total += 1
    # prompt = SYSTEM_PROMPT + format_fewshot(fewshot, mmlu_item)
    prompt = SYSTEM_PROMPT + mmlu_item.format_instruct_q()
    tokens = llama.tokenizer.encode(prompt, **encode_kwargs)
    tokens = torch.tensor(tokens, device=device).reshape(1, -1)
    attn_mask = llama._build_attn_mask(tokens.shape[-1], None)
    print(llama.tokenizer.decode(tokens[0].tolist()), end="")
    for chunk in llama.generate(tokens, attn_mask, sampler="topk_greedy", temp=0, k=4):
      pred = chunk["choices"][0]["delta"]["content"].strip()
      answer = chr(mmlu_item.answer + ord("A"))
      print(f"({pred},{answer})", end="", flush=True)
      result = {
        "prompt": prompt,
        "pred": pred,
        "answer": answer,
        "logprobs": [
          {"token": tp["token"], "logprob": tp["logprob"]}
          for tp in chunk["top_predictions"]
        ],
      }
      correct += pred == answer
      break
  print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")

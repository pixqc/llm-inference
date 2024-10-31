import csv
import json
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional

from src.torch_main import LLAMA_1B_PARAMS, Llama


class MMLU(NamedTuple):
  question: str
  choices: List[str]
  answer: int

  def format(self) -> tuple[str, str]:
    prompt = (
      "Given the following question and four candidate answers (A, B, C and D), choose the best answer."
      f"\nQuestion: {self.question}\n"
      + "\n".join(f"{c}. {a}" for c, a in zip("ABCD", self.choices))
      + '\nYour response should end with "The best answer is [the_answer_letter]" where the [the_answer_letter] is one of A, B, C or D.'
    )
    return prompt, chr(self.answer + ord("A"))


def load_mmlu(path: Path) -> Iterator[MMLU]:
  return (
    MMLU(row[0], row[1:5], ord(row[5]) - ord("A"))
    for csv_file in path.glob("*.csv")
    for row in csv.reader(open(csv_file))
    if len(row) == 6
  )


def format_fewshot(question: str, examples: Optional[List[tuple[str, str]]]) -> str:
  if examples:
    examples_text = "\n\n".join(f"{q}\nThe best answer is {a}." for q, a in examples)
    return f"{examples_text}\n\n{question}"
  return question


if __name__ == "__main__":
  fewshot = [ex.format() for ex in list(load_mmlu(Path("evals/mmlu/val")))[:2]]
  mmlu = [ex.format() for ex in load_mmlu(Path("evals/mmlu/test"))]

  is_instruct = True
  weight_path = f"src/model/1B{'-Instruct' if is_instruct else ''}"
  llama = Llama(is_instruct, LLAMA_1B_PARAMS, weight_path, "src/tokenizer.model")

  correct = total = 0
  with open("evals/eval_mmlu.jsonl", "a") as f:
    for question, answer in mmlu:
      total += 1
      prompt = format_fewshot(question, fewshot)
      tokens, attn_mask = llama.tokenize(
        [prompt], format_instruct=True, postfix="The best answer is"
      )
      print(llama.detokenize(tokens[0]), end="")
      for chunk in llama.generate(
        tokens, attn_mask, sampler="topk_greedy", temp=0, k=4
      ):
        pred = chunk["choices"][0]["delta"]["content"].strip()
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
        f.write(f"{json.dumps(result)}\n")
        break
  print(f"\nMMLU Accuracy: {correct/total:.2%}")

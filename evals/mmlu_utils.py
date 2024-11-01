import csv
from pathlib import Path
from typing import Iterator, List, NamedTuple


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

# https://github.com/xjdr-alt/llama_inference/blob/main/llama_inference/tokenizer.py

import os
from pathlib import Path
from typing import (
  AbstractSet,
  Collection,
  Dict,
  Iterator,
  List,
  Literal,
  Optional,
  Sequence,
  Union,
  cast,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe

TIKTOKEN_MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACES_CHARS = 25_000


class Tokenizer:
  special_tokens: Dict[str, int]
  num_reserved_special_tokens = 256
  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

  def __init__(self, model_path: str):
    assert os.path.isfile(model_path), model_path
    mergeable_ranks = load_tiktoken_bpe(model_path)
    num_base_tokens = len(mergeable_ranks)
    special_tokens = [
      "<|begin_of_text|>",
      "<|end_of_text|>",
      "<|reserved_special_token_0|>",
      "<|reserved_special_token_1|>",
      "<|finetune_right_pad_id|>",
      "<|step_id|>",
      "<|start_header_id|>",
      "<|end_header_id|>",
      "<|eom_id|>",  # end of message
      "<|eot_id|>",  # end of turn
      "<|python_tag|>",
    ]
    reserved_tokens = [
      f"<|reserved_special_token_{2 + i}|>"
      for i in range(self.num_reserved_special_tokens - len(special_tokens))
    ]
    special_tokens = special_tokens + reserved_tokens

    self.special_tokens = {
      token: num_base_tokens + i for i, token in enumerate(special_tokens)
    }
    self.model = tiktoken.Encoding(
      name=Path(model_path).name,
      pat_str=self.pat_str,
      mergeable_ranks=mergeable_ranks,
      special_tokens=self.special_tokens,
    )

    self.n_words: int = num_base_tokens + len(special_tokens)
    # BOS / EOS token IDs
    self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
    self.eos_id: int = self.special_tokens["<|end_of_text|>"]
    self.eot_id: int = self.special_tokens["<|eot_id|>"]
    self.eom_id: int = self.special_tokens["<|eom_id|>"]
    self.python_tag_id = self.special_tokens["<|python_tag|>"]
    self.pad_id: int = self.special_tokens["<|finetune_right_pad_id|>"]
    self.stop_tokens = [
      self.special_tokens["<|eom_id|>"],
      self.special_tokens["<|eot_id|>"],
    ]

  def encode(
    self,
    s: str,
    *,
    bos: bool,
    eos: bool,
    allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
    disallowed_special: Union[Literal["all"], Collection[str]] = (),
  ) -> List[int]:
    if allowed_special is None:
      allowed_special = set()
    assert isinstance(s, str)

    substrs = (
      substr
      for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
      for substr in self._split_whitespaces_or_nonwhitespaces(
        s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
      )
    )
    t: List[int] = []
    for substr in substrs:
      t.extend(
        self.model.encode(
          substr,
          allowed_special=allowed_special,
          disallowed_special=disallowed_special,
        )
      )
    if bos:
      t.insert(0, self.bos_id)
    if eos:
      t.append(self.eos_id)
    return t

  def decode(self, t: Sequence[int]) -> str:
    return self.model.decode(cast(List[int], t))

  @staticmethod
  def _split_whitespaces_or_nonwhitespaces(
    s: str, max_consecutive_slice_len: int
  ) -> Iterator[str]:
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0

    for i in range(len(s)):
      is_now_space = s[i].isspace()

      if current_slice_is_space ^ is_now_space:
        current_slice_len = 1
        current_slice_is_space = is_now_space
      else:
        current_slice_len += 1
        if current_slice_len > max_consecutive_slice_len:
          yield s[slice_start:i]
          slice_start = i
          current_slice_len = 1
    yield s[slice_start:]

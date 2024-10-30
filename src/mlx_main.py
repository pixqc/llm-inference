import hashlib
import time
from pathlib import Path
from typing import List, Literal, NamedTuple, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from src.tokenizer import Tokenizer

Sampler = Literal["greedy", "topk", "topp", "topk_greedy", "minp"]


class ModelParams(NamedTuple):
  n_layers: int
  n_local_heads: int
  n_local_kv_heads: int
  head_dim: int
  max_seqlen: int
  rope_theta: float
  use_scaled_rope: bool


LLAMA_1B_PARAMS = ModelParams(
  n_layers=16,
  n_local_heads=32,
  n_local_kv_heads=8,
  head_dim=64,
  max_seqlen=4096,
  rope_theta=500000.0,
  use_scaled_rope=True,
)


class LayerWeights(NamedTuple):
  wq: mx.array
  wk: mx.array
  wv: mx.array
  wo: mx.array
  w1: mx.array
  w2: mx.array
  w3: mx.array
  ffn_norm: mx.array
  attention_norm: mx.array


class XfmrWeights(NamedTuple):
  tok_embeddings: mx.array
  norm: mx.array
  output: mx.array
  layer_weights: List[LayerWeights]


class Rope:
  def __init__(
    self,
    dim: int,
    max_seqlen: int,
    theta: float = 500000.0,
    use_scaled: bool = True,
    dtype=mx.float32,
  ):
    self.dim = dim
    self.max_seqlen = max_seqlen
    self.theta = theta
    self.use_scaled = use_scaled
    self.dtype = dtype
    self.freqs_sin, self.freqs_cos = self._precompute_freqs_cis()

  def _precompute_freqs_cis(self):
    real = lambda x: mx.view(x, self.dtype)[..., ::2]
    imag = lambda x: mx.view(x, self.dtype)[..., 1::2]
    freqs = 1.0 / (
      self.theta
      ** (mx.arange(0, self.dim, 2)[: (self.dim // 2)].astype(self.dtype) / self.dim)
    )
    if self.use_scaled:
      pass  # No scaling implemented yet
    t = mx.arange(self.max_seqlen, dtype=self.dtype)
    freqs = mx.outer(t, freqs)
    freqs = mx.exp(1j * freqs)
    return imag(freqs), real(freqs)

  # TODO: why is there a seqlen here
  def apply_rotary_emb(self, xq, xk, start_pos: int, seqlen: Optional[int] = None):
    if seqlen is None:
      freqs_sin = self.freqs_sin[start_pos : start_pos + 1]
      freqs_cos = self.freqs_cos[start_pos : start_pos + 1]
    else:
      freqs_sin = self.freqs_sin[:seqlen]
      freqs_cos = self.freqs_cos[:seqlen]
    xq_r, xq_i = xq[..., ::2], xq[..., 1::2]
    xk_r, xk_i = xk[..., ::2], xk[..., 1::2]
    freqs_sin = freqs_sin[None, :, None, :]
    freqs_cos = freqs_cos[None, :, None, :]
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    xq_out = mx.stack([xq_out_r, xq_out_i], axis=-1).reshape(xq.shape)
    xk_out = mx.stack([xk_out_r, xk_out_i], axis=-1).reshape(xk.shape)
    return xq_out, xk_out


class KVCache(NamedTuple):
  k: mx.array
  v: mx.array

  @classmethod
  def init(cls, bsz: int, model_params: ModelParams) -> "KVCache":
    return cls(
      k=mx.zeros(
        (
          model_params.n_layers,
          bsz,
          model_params.max_seqlen,
          model_params.n_local_kv_heads,
          model_params.head_dim,
        )
      ),
      v=mx.zeros(
        (
          model_params.n_layers,
          bsz,
          model_params.max_seqlen,
          model_params.n_local_kv_heads,
          model_params.head_dim,
        )
      ),
    )

  def update(
    self,
    layer_idx: int,
    bsz: int,
    cur_pos: int,
    xk: mx.array,
    xv: mx.array,
    n_rep: int,
  ):
    if cur_pos == 0:
      self.k[layer_idx, :bsz, : xk.shape[1]] = xk
      self.v[layer_idx, :bsz, : xv.shape[1]] = xv
    else:
      self.k[layer_idx, :bsz, cur_pos - 1] = xk.squeeze(1)
      self.v[layer_idx, :bsz, cur_pos - 1] = xv.squeeze(1)
      xk = self.k[layer_idx, :bsz, :cur_pos]
      xv = self.v[layer_idx, :bsz, :cur_pos]
    xk = mx.repeat(xk, n_rep, axis=2)
    xv = mx.repeat(xv, n_rep, axis=2)
    return xk, xv, self


class Llama:
  def _load_weights(self, dir: str, n_layers: int = 16):
    """
    The model architecture for Llama 3.2 1B:
    LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(128256, 2048)
        (layers): ModuleList(
          (0-15): 16 x LlamaDecoderLayer(
            (self_attn): LlamaSdpaAttention(
              (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
              (k_proj): Linear(in_features=2048, out_features=512, bias=False)
              (v_proj): Linear(in_features=2048, out_features=512, bias=False)
              (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
              (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): LlamaMLP(
              (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
              (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
              (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
            (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
          )
        )
        (norm): LlamaRMSNorm((2048,), eps=1e-05)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
    )
    """
    w = {}
    layer_weights = []
    for file in Path(dir).glob("*.npy"):
      name = ".".join(str(file).split("/")[-1].split(".")[:-1])
      w[name] = mx.load(str(file), "npy")

    for i in range(n_layers):
      layer_weights.append(
        LayerWeights(
          wq=w[f"layers.{i}.attention.wq.weight"],
          wk=w[f"layers.{i}.attention.wk.weight"],
          wv=w[f"layers.{i}.attention.wv.weight"],
          wo=w[f"layers.{i}.attention.wo.weight"],
          w1=w[f"layers.{i}.feed_forward.w1.weight"],
          w2=w[f"layers.{i}.feed_forward.w2.weight"],
          w3=w[f"layers.{i}.feed_forward.w3.weight"],
          ffn_norm=w[f"layers.{i}.ffn_norm.weight"],
          attention_norm=w[f"layers.{i}.attention_norm.weight"],
        )
      )

    xfmr_weights = XfmrWeights(
      tok_embeddings=w["tok_embeddings.weight"],
      norm=w["norm.weight"],
      output=w["output.weight"],
      layer_weights=layer_weights,
    )

    return xfmr_weights

  def __init__(
    self,
    is_instruct: bool,
    model_params: ModelParams,
    weights_path: str,
    tokenizer_path: str,
    bsz: int = 1,
  ):
    self.params = model_params
    self.is_instruct = is_instruct
    self.weights = self._load_weights(weights_path)
    self.tokenizer = Tokenizer(tokenizer_path)
    self.xfmr = Transformer(model_params, self.weights)
    self.kvcache = KVCache.init(bsz, model_params)

  def _get_candidates(
    self, logits: mx.array, _type: str, temp=0.7, **kwargs
  ) -> Tuple[mx.array, mx.array]:
    if temp == 0:
      temp = 1e-4
    last_logits = logits[:, -1]
    sorted_indices = mx.argsort(last_logits, axis=-1)[:, ::-1]
    sorted_logits = mx.take_along_axis(last_logits, sorted_indices, axis=-1)
    logprobs = nn.log_softmax(sorted_logits / temp, axis=-1)

    if _type == "greedy":
      cutoff = 1
    elif _type == "topk" or _type == "topk_greedy":
      if "k" not in kwargs:
        raise ValueError("'k' parameter is required for topk sampling")
      cutoff = kwargs["k"]
    elif _type == "minp":
      if "p" not in kwargs:
        raise ValueError("'p' parameter is required for minp sampling")
      p = kwargs["p"]
      probs = mx.exp(logprobs)
      cutoff = mx.max((probs > p).sum(axis=-1)).item()
    elif _type == "topp":
      if "p" not in kwargs:
        raise ValueError("'p' parameter is required for topp sampling")
      p = kwargs["p"]
      probs = mx.exp(logprobs)
      probs_cumsum = mx.cumsum(probs, axis=-1)
      cutoff = mx.max(mx.sum(probs_cumsum <= p, axis=-1)).item()
    else:
      raise ValueError(f"Unknown sampling type: {_type}")

    if (_type == "minp" or _type == "topp") and cutoff == 0:
      cutoff = 50  # fallback to topk
    candidates = sorted_indices[:, :cutoff]
    logprobs = logprobs[:, :cutoff]
    logprobs = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
    return candidates, logprobs

  def _pad_batch(self, batch_tokens: List[mx.array], pad_id: int):
    longest_seqlen = max(len(tokens) for tokens in batch_tokens)
    for i in range(len(batch_tokens)):
      pad = mx.full((longest_seqlen - len(batch_tokens[i]),), pad_id)
      batch_tokens[i] = mx.concatenate([pad, batch_tokens[i]])
    padded = mx.stack(batch_tokens, axis=0)
    return padded, (padded == pad_id)

  def _build_attn_mask(self, seqlen: int, pad_mask: Optional[mx.array]):
    mask = mx.zeros((seqlen, seqlen), dtype=mx.float32)
    if seqlen > 1:
      mask = mx.full((seqlen, seqlen), float("-inf"))
      mask = mx.triu(mask, k=1)
    if pad_mask is None:
      return mask
    src_mask = pad_mask[:, :, None]
    target_mask = pad_mask[:, None, :]
    pad_mask = src_mask | target_mask
    mask = mx.broadcast_to(mask[None, :, :], (pad_mask.shape[0], seqlen, seqlen))  # noqa
    mask = mx.where(pad_mask, float("-inf"), mask)[:, None, :, :]
    return mask

  def _random_sample(self, candidates: mx.array, logprobs: mx.array, sampler: Sampler):
    if sampler == "topk_greedy" or sampler == "greedyy":
      return candidates[:, 0].reshape(-1, 1), logprobs[:, 0]
    idx = mx.random.categorical(logprobs)
    batch_indices = mx.arange(candidates.shape[0])
    return candidates[batch_indices, idx].reshape(-1, 1), logprobs[batch_indices, idx]

  def _format_instruct(self, prompt: str, system_prompt: Optional[str] = None):
    return (
      f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
      f"{ '' if system_prompt is None else system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
      f"{ prompt }<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

  def tokenize(
    self,
    prompts: list[str],
    format_instruct: bool,
    postfix: Optional[str] = None,
  ) -> Tuple[mx.array, mx.array]:
    """
    Postfix is for putting words in the assistant's mouth
    eg. "Let's think step by step."
    """
    assert isinstance(prompts, list), "prompts must be a list of strings"
    encode_kwargs = {"bos": False, "eos": False, "allowed_special": "all"}
    if format_instruct and self.is_instruct:
      prompts = [self._format_instruct(p) for p in prompts]
    if postfix is not None:
      prompts = [p + postfix for p in prompts]
    tokens = [mx.array(self.tokenizer.encode(p, **encode_kwargs)) for p in prompts]
    tokens, pad_mask = self._pad_batch(tokens, self.tokenizer.eos_id)
    attn_mask = self._build_attn_mask(tokens.shape[-1], pad_mask)  # type: ignore
    return tokens, attn_mask

  def detokenize(self, tokens: mx.array) -> List[str]:
    assert len(tokens.shape) == 1, "tokens must be shape (seqlen)"
    return self.tokenizer.decode(tokens.tolist())  # type: ignore

  def generate(
    self,
    tokens: mx.array,
    attn_mask: mx.array,
    sampler: Sampler,
    temp: float,
    **kwargs,
  ):
    assert len(tokens.shape) == 2, "tokens must be shape (bsz, seqlen)"
    STOP_TOKENS = [128001, 128008, 128009]
    chat_id = hashlib.md5(str(time.time()).encode("utf-8")).hexdigest()
    cur_pos = 0

    while cur_pos < self.params.max_seqlen:
      x = tokens if cur_pos == 0 else tokens[:, -1:]
      attn_mask = attn_mask if cur_pos == 0 else mx.array([0])
      logits, self.kvcache = self.xfmr(x, self.kvcache, cur_pos, attn_mask)  # type: ignore
      candidates, logprobs = self._get_candidates(logits, sampler, temp=temp, **kwargs)
      next_token, _ = self._random_sample(candidates, logprobs, sampler)

      if cur_pos == 0:
        cur_pos = tokens.shape[-1]
      else:
        cur_pos += 1

      yield {
        "id": f"chatcmpl-{chat_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "Llama-3.2-1B-Instruct" if self.is_instruct else "Llama-3.2-1B",
        "choices": [
          {
            "index": 0,
            "delta": {"content": self.tokenizer.decode(next_token[0].tolist())},  # type: ignore
            "finish_reason": "stop" if next_token[0] in STOP_TOKENS else None,
          }
        ],
        "current_token_position": cur_pos,
        "top_predictions": [
          {"token": self.tokenizer.decode([token]), "logprob": logprob}
          for token, logprob in zip(candidates[0].tolist(), logprobs[0].tolist())  # type: ignore
        ],
      }
      if next_token[0] in STOP_TOKENS:
        break
      tokens = mx.concatenate([tokens, next_token], axis=1)


class AttentionBlock:
  def __init__(
    self, layer_index: int, params: ModelParams, layer_weights: LayerWeights
  ):
    self.idx = layer_index
    self.params = params
    self.weights = layer_weights

  def __call__(
    self,
    x: mx.array,
    rope: Rope,
    kvcache: KVCache,
    cur_pos: int,
    mask: mx.array,
  ):
    x = mx.fast.rms_norm(x, self.weights.attention_norm, 1e-6)
    bsz, seqlen, _ = x.shape
    hc, hsz, gsz = (
      self.params.n_local_heads,
      self.params.head_dim,
      self.params.n_local_kv_heads,
    )
    xq = (x @ self.weights.wq.T).reshape(bsz, -1, hc, hsz)
    xk = (x @ self.weights.wk.T).reshape(bsz, -1, gsz, hsz)
    xv = (x @ self.weights.wv.T).reshape(bsz, -1, gsz, hsz)
    xq, xk = rope.apply_rotary_emb(xq, xk, cur_pos, seqlen if cur_pos == 0 else None)
    xk, xv, kvcache = kvcache.update(self.idx, bsz, cur_pos, xk, xv, hc // gsz)
    scores = mx.einsum("bihd,bjhd->bhij", xq, xk)
    scores = (scores + mask) / mx.sqrt(mx.array(hsz))
    scores = mx.softmax(scores, axis=-1)
    out = mx.einsum("bhij,bjhk->bihk", scores, xv)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    return out @ self.weights.wo.T, kvcache


class Transformer:
  def _ffw(self, x: mx.array, layer_weights: LayerWeights):
    x = mx.fast.rms_norm(x, layer_weights.ffn_norm, 1e-6)
    return (
      nn.silu(x @ layer_weights.w1.T) * (x @ layer_weights.w3.T) @ layer_weights.w2.T
    )

  def __init__(self, model_params: ModelParams, weights: XfmrWeights):
    self.params = model_params
    self.weights = weights
    self.rope = Rope(
      dim=model_params.head_dim,
      max_seqlen=model_params.max_seqlen,
      theta=model_params.rope_theta,
      use_scaled=model_params.use_scaled_rope,
    )
    self.attns = [
      AttentionBlock(i, model_params, layer_weights)
      for i, layer_weights in enumerate(self.weights.layer_weights)
    ]

  def __call__(
    self, tokens: mx.array, kvcache: KVCache, cur_pos: int, attn_mask: mx.array
  ):
    x = self.weights.tok_embeddings[tokens]
    x = x[None, :] if len(x.shape) < 3 else x

    for i in range(self.params.n_layers):
      attn_out, kvcache = self.attns[i](
        x,
        self.rope,
        kvcache,
        cur_pos,
        attn_mask,
      )
      x = x + attn_out
      x = x + self._ffw(x, self.weights.layer_weights[i])
    x = mx.fast.rms_norm(x, self.weights.norm, 1e-6)
    logits = x @ self.weights.output.T
    return logits, kvcache


if __name__ == "__main__":
  is_instruct = True
  weight_path, tok_path = "src/model/1B", "src/tokenizer.model"
  weight_path = weight_path + "-Instruct" if is_instruct else weight_path
  prompts = ["What is the capital of France?"]
  llama = Llama(is_instruct, LLAMA_1B_PARAMS, weight_path, tok_path, len(prompts))
  tokens, attn_mask = llama.tokenize(prompts, format_instruct=True)
  print(llama.detokenize(tokens[0]))
  for chunk in llama.generate(tokens, attn_mask, sampler="topk", temp=0.6, k=5):
    print(chunk["choices"][0]["delta"]["content"], end="", flush=True)

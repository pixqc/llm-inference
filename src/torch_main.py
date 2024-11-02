import hashlib
import json
import time
from dataclasses import dataclass
from typing import List, Literal, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from src.tokenizer import Tokenizer

device = torch.device("cpu")
if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.cuda.is_available():
  device = torch.device("cuda")
print(f"using device: {device}")

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E


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
  wq: torch.Tensor
  wk: torch.Tensor
  wv: torch.Tensor
  wo: torch.Tensor
  w1: torch.Tensor
  w2: torch.Tensor
  w3: torch.Tensor
  ffn_norm: torch.Tensor
  attention_norm: torch.Tensor


class XfmrWeights(NamedTuple):
  tok_embeddings: torch.Tensor
  norm: torch.Tensor
  output: torch.Tensor
  layer_weights: List[LayerWeights]


class Rope:
  @staticmethod
  def precompute_freqs_cis(
    dim: int,
    max_seqlen: int,
    theta: float = 500000.0,
    use_scaled: bool = True,
    dtype=torch.float32,
  ) -> torch.Tensor:
    freqs = 1.0 / (
      theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim)
    )
    if use_scaled:
      pass
    t = torch.arange(max_seqlen, dtype=dtype, device=device).unsqueeze(1)
    freqs = freqs.unsqueeze(0)
    freqs = t * freqs
    return torch.exp(1j * freqs)

  @staticmethod
  def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(
      *xq_out.shape[:-1], -1
    )
    xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(
      *xk_out.shape[:-1], -1
    )
    return xq_out.to(dtype), xk_out.to(dtype)


class KVCache(NamedTuple):
  k: torch.Tensor
  v: torch.Tensor

  @classmethod
  def init(cls, bsz: int, model_params: ModelParams) -> "KVCache":
    return cls(
      k=torch.zeros(
        (
          model_params.n_layers,
          bsz,
          model_params.max_seqlen,
          model_params.n_local_kv_heads,
          model_params.head_dim,
        ),
        device=device,
        dtype=torch.bfloat16,
      ),
      v=torch.zeros(
        (
          model_params.n_layers,
          bsz,
          model_params.max_seqlen,
          model_params.n_local_kv_heads,
          model_params.head_dim,
        ),
        device=device,
        dtype=torch.bfloat16,
      ),
    )

  def update(
    self,
    layer_idx: int,
    bsz: int,
    cur_pos: int,
    xk: torch.Tensor,
    xv: torch.Tensor,
    n_rep: int,
  ):
    if cur_pos == 0:
      self.k[layer_idx, :bsz, : xk.shape[1]].copy_(xk.to(device))
      self.v[layer_idx, :bsz, : xv.shape[1]].copy_(xv.to(device))
    else:
      self.k[layer_idx, :bsz, cur_pos - 1].copy_(xk.squeeze(1).to(device))
      self.v[layer_idx, :bsz, cur_pos - 1].copy_(xv.squeeze(1).to(device))
      xk = self.k[layer_idx, :bsz, :cur_pos]
      xv = self.v[layer_idx, :bsz, :cur_pos]
    xk = xk.repeat_interleave(n_rep, dim=2)
    xv = xv.repeat_interleave(n_rep, dim=2)
    return xk, xv, self


class AttentionBlock:
  def __init__(
    self, layer_index: int, params: ModelParams, layer_weights: LayerWeights
  ):
    self.idx = layer_index
    self.params = params
    self.weights = layer_weights

  def __call__(
    self,
    x: torch.Tensor,
    kvcache: KVCache,
    cur_pos: int,
    mask: torch.Tensor,
    freqs_cis: torch.Tensor,
  ):
    x = F.rms_norm(x, x.shape[-1:], self.weights.attention_norm, 1e-6)
    bsz, _, _ = x.shape
    hc, hsz, gsz = (
      self.params.n_local_heads,
      self.params.head_dim,
      self.params.n_local_kv_heads,
    )
    xq = (x @ self.weights.wq.T).reshape(bsz, -1, hc, hsz)
    xk = (x @ self.weights.wk.T).reshape(bsz, -1, gsz, hsz)
    xv = (x @ self.weights.wv.T).reshape(bsz, -1, gsz, hsz)
    xq, xk = Rope.apply_rotary_emb(xq, xk, freqs_cis)
    xk, xv, kvcache = kvcache.update(self.idx, bsz, cur_pos, xk, xv, hc // gsz)
    _scores = torch.einsum("bihd,bjhd->bhij", xq, xk) / (hsz**0.5)
    scores = _scores + mask
    scores = F.softmax(scores, dim=-1).to(torch.bfloat16)
    out = torch.einsum("bhij,bjhk->bihk", scores, xv)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    return out @ self.weights.wo.T, kvcache, _scores


class Transformer:
  def _ffw(self, x: torch.Tensor, lw: LayerWeights):
    x = F.rms_norm(x, x.shape[-1:], lw.ffn_norm, 1e-6)
    return F.silu(x @ lw.w1.T) * (x @ lw.w3.T) @ lw.w2.T

  def __init__(self, model_params: ModelParams, weights: XfmrWeights):
    self.params = model_params
    self.weights = weights
    self.attns = [
      AttentionBlock(i, model_params, layer_weights)
      for i, layer_weights in enumerate(self.weights.layer_weights)
    ]

  def __call__(
    self,
    tokens: torch.Tensor,
    kvcache: KVCache,
    cur_pos: int,
    attn_mask: torch.Tensor,
    freqs_cis: torch.Tensor,
  ):
    x = self.weights.tok_embeddings[tokens]
    x = x[None, :] if len(x.shape) < 3 else x

    for i in range(self.params.n_layers):
      attn_out, kvcache, scores = self.attns[i](
        x,
        kvcache,
        cur_pos,
        attn_mask,
        freqs_cis,
      )
      x = x + attn_out
      x = x + self._ffw(x, self.weights.layer_weights[i])
    x = F.rms_norm(x, x.shape[-1:], self.weights.norm, 1e-6)
    logits = x @ self.weights.output.T
    return logits, kvcache, scores


@dataclass
class SamplerConfig:
  low_logits_entropy_threshold: float = 0.8
  medium_logits_entropy_threshold: float = 1.7
  high_logits_entropy_threshold: float = 2.5

  # Varentropy thresholds
  low_logits_varentropy_threshold: float = 0.4
  high_logits_varentropy_threshold: float = 1.2

  # Attention metrics thresholds
  low_attention_entropy_threshold: float = 1.0
  high_attention_entropy_threshold: float = 2.0
  low_attention_varentropy_threshold: float = 0.3
  high_attention_varentropy_threshold: float = 0.8

  # Base sampling parameters
  temperature: float = 0.8
  top_k: int = 40
  top_p: float = 0.9
  min_probability: float = 0.05

  # Adaptive sampling parameters
  target_ce_alpha: float = 1.2
  target_ce_beta: float = 2.0
  number_of_adaptive_samples: int = 5
  adaptive_temperature_logits_coefficient: float = 0.3
  adaptive_temperature_attention_coefficient: float = 0.2
  adaptive_top_p_coefficient: float = 0.15
  adaptive_min_p_coefficient: float = 0.2

  # Scoring coefficients
  adaptive_score_logits_entropy_coefficient: float = 1.0
  adaptive_score_attention_entropy_coefficient: float = 0.8
  adaptive_score_logits_varentropy_coefficient: float = 0.6
  adaptive_score_attention_varentropy_coefficient: float = 0.4

  # High entropy adjustments
  high_entropy_attention_offset: float = 0.2
  high_entropy_attention_coefficient: float = 0.3
  high_entropy_varentropy_attention_offset: float = 0.3
  high_entropy_varentropy_attention_coefficient: float = 0.25


class Sampler:
  Type = Literal["greedy", "topk", "topp", "topk_greedy", "minp", "entropix"]

  @staticmethod
  def get_candidates(
    logits: torch.Tensor, _type: Type, temp=0.7, **kwargs
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    if temp == 0:
      temp = 1e-9
    last_logits = logits[:, -1]

    sorted_logits, sorted_indices = torch.sort(last_logits, dim=-1, descending=True)
    logprobs = torch.nn.functional.log_softmax(sorted_logits / temp, dim=-1)
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
      probs = torch.exp(logprobs)
      cutoff = (probs > p).sum(dim=-1).max().item()
    elif _type == "topp":
      if "p" not in kwargs:
        raise ValueError("'p' parameter is required for topp sampling")
      p = kwargs["p"]
      probs = torch.exp(logprobs)
      probs_cumsum = torch.cumsum(probs, dim=-1)
      cutoff = (probs_cumsum <= p).sum(dim=-1).max().item()
    elif _type == "entropix":
      if "metrics" not in kwargs:
        raise ValueError("'metrics' parameter is required for entropix sampling")
      config = SamplerConfig()
      metrics = kwargs["metrics"]
      ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
      attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]

      LELV = (
        ent < config.low_logits_entropy_threshold
        and vent < config.low_logits_varentropy_threshold
        and attn_ent < config.low_attention_entropy_threshold
        and attn_vent < config.low_attention_varentropy_threshold
      )
      HELV = (
        ent > config.high_logits_entropy_threshold
        and vent < config.low_logits_varentropy_threshold
        and attn_ent < config.low_attention_entropy_threshold
        and attn_vent < config.low_attention_varentropy_threshold
      )
      LEHV = (
        ent < config.high_logits_entropy_threshold
        and vent > config.high_logits_varentropy_threshold
        and attn_ent < config.low_attention_entropy_threshold
        and attn_vent > config.high_attention_varentropy_threshold
      )
      HEHV = (
        ent > config.medium_logits_entropy_threshold
        and vent > config.high_logits_varentropy_threshold
        and attn_ent > config.high_attention_entropy_threshold
        and attn_vent > config.high_attention_varentropy_threshold
      )

      if LELV:  # argmax
        # Low Entropy, Low Varentropy: "flowing with unspoken intent"
        cutoff = 1
      elif HELV:
        # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
        clarifying_question_token = 2564
        cutoff = 1
        candidates = [[clarifying_question_token]]
        logprobs = torch.tensor([[0.0]])
        print("helv")
      elif LEHV:
        # Low Entropy, High Varentropy: "exploring forks in the path"
        probs = torch.exp(logprobs)
        topk_cutoff = config.top_k
        minp_cutoff = (probs > config.min_probability).sum(dim=-1).max().item()
        sorted_probs = torch.sort(probs, dim=-1, descending=True)[0]
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        topp_cutoff = (cumsum_probs <= config.top_p).sum(dim=-1).max().item()
        cutoff = min(topk_cutoff, minp_cutoff, topp_cutoff)
        temp = config.temperature
        print("lehv")
      elif HEHV:
        # High Entropy, High Varentropy: "resampling in the mist"
        temp = (
          config.high_entropy_varentropy_attention_offset
          + config.high_entropy_varentropy_attention_coefficient * attn_vent
        )
        top_p_adj = torch.maximum(
          torch.tensor(0.5),
          config.top_p - config.high_entropy_attention_coefficient * attn_ent,
        )
        probs = torch.exp(logprobs)
        topk_cutoff = config.top_k
        minp_cutoff = (probs > config.min_probability).sum(dim=-1).max().item()
        sorted_probs = torch.sort(probs, dim=-1, descending=True)[0]
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        topp_cutoff = (cumsum_probs <= top_p_adj).sum(dim=-1).max().item()
        cutoffs = [topk_cutoff, minp_cutoff, topp_cutoff]
        cutoff = min([cutoff for cutoff in cutoffs if cutoff > 0])
        print("hehv")
      else:
        # should be adaptive sampling but argmax for now
        cutoff = 1
    else:
      raise ValueError(f"Unknown sampling type: {_type}")

    if (_type == "minp" or _type == "topp") and cutoff == 0:
      cutoff = 50  # fallback to topk
    candidates = sorted_indices[:, :cutoff]
    logprobs = logprobs[:, :cutoff]
    logprobs = logprobs - torch.logsumexp(logprobs, dim=-1, keepdim=True)
    return candidates, logprobs

  @staticmethod
  def random_choice(candidates: torch.Tensor, logprobs: torch.Tensor, sampler: Type):
    if sampler == "topk_greedy" or sampler == "greedy":
      return candidates[:, 0].view(-1, 1), logprobs[:, 0]
    idx = torch.multinomial(torch.exp(logprobs), num_samples=1).squeeze()
    batch_indices = torch.arange(candidates.shape[0])
    return candidates[batch_indices, idx].view(-1, 1), logprobs[batch_indices, idx]

  @staticmethod
  def calculate_ent_vent(
    logits: torch.Tensor, axis: int = -1
  ) -> tuple[torch.Tensor, torch.Tensor]:
    logprobs = F.log_softmax(logits, dim=axis)
    probs = torch.exp(logprobs)
    entropy = -torch.sum(probs * logprobs, dim=axis) / LN_2
    varentropy = torch.sum(
      probs * (logprobs / LN_2 + entropy.unsqueeze(-1)) ** 2, dim=axis
    )
    return entropy, varentropy

  @staticmethod
  def calculate_metrics(
    logits: torch.Tensor, attn_scores: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    entropy, varentropy = Sampler.calculate_ent_vent(logits)
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_entropy = -torch.sum(
      attn_probs * torch.log2(torch.clamp(attn_probs, 1e-10, 1.0)), dim=-1
    )
    attn_varentropy = torch.var(attn_entropy, dim=1)
    return {
      "logits_entropy": torch.mean(entropy),
      "logits_varentropy": torch.mean(varentropy),
      "attn_entropy": torch.mean(attn_entropy),
      "attn_varentropy": torch.mean(attn_varentropy),
    }


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
    weights = load_file(f"{dir}/model.safetensors")
    layer_weights = []
    for i in range(n_layers):
      layer_weights.append(
        LayerWeights(
          wq=weights[f"layers.{i}.attention.wq.weight"].to(device),
          wk=weights[f"layers.{i}.attention.wk.weight"].to(device),
          wv=weights[f"layers.{i}.attention.wv.weight"].to(device),
          wo=weights[f"layers.{i}.attention.wo.weight"].to(device),
          w1=weights[f"layers.{i}.feed_forward.w1.weight"].to(device),
          w2=weights[f"layers.{i}.feed_forward.w2.weight"].to(device),
          w3=weights[f"layers.{i}.feed_forward.w3.weight"].to(device),
          ffn_norm=weights[f"layers.{i}.ffn_norm.weight"].to(device),
          attention_norm=weights[f"layers.{i}.attention_norm.weight"].to(device),
        )
      )
    xfmr_weights = XfmrWeights(
      tok_embeddings=weights["tok_embeddings.weight"].to(device),
      norm=weights["norm.weight"].to(device),
      output=weights["output.weight"].to(device),
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
    self.stop_tokens = [128001, 128008, 128009]
    self.xfmr = Transformer(model_params, self.weights)
    self.kvcache = KVCache.init(bsz, model_params)
    self.freqs_cis_all = Rope.precompute_freqs_cis(
      model_params.head_dim,
      model_params.max_seqlen,
      model_params.rope_theta,
      model_params.use_scaled_rope,
    )

  @staticmethod
  def _pad_batch(batch_tokens: List[torch.Tensor], pad_id: int):
    longest_seqlen = max(len(tokens) for tokens in batch_tokens)
    padded_tokens = []
    for tokens in batch_tokens:
      padding = torch.full(
        (longest_seqlen - len(tokens),),
        pad_id,
        dtype=tokens.dtype,
        device=tokens.device,
      )
      padded_tokens.append(torch.cat([padding, tokens]))
    padded = torch.stack(padded_tokens).to(device=device)
    return padded, (padded == pad_id).to(device=device)

  @staticmethod
  def _build_attn_mask(seqlen: int, pad_mask: Optional[torch.Tensor]):
    mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    if pad_mask is not None:
      pad_mask = pad_mask[:, :, None] | pad_mask[None, :, :]
      mask = torch.where(pad_mask, float("-inf"), mask)[:, None, :, :]
    return mask

  @staticmethod
  def _format_instruct(prompt: str, system_prompt: Optional[str] = None):
    return (
      f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
      f"{ '' if system_prompt is None else system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
      f"{ prompt }<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

  # TODO: should tokenization happen on callsite?
  def tokenize(
    self,
    prompts: list[str],
    format_instruct: bool,
    postfix: Optional[str] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
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
    tokens = [torch.tensor(self.tokenizer.encode(p, **encode_kwargs)) for p in prompts]
    tokens, pad_mask = self._pad_batch(tokens, self.tokenizer.eos_id)
    attn_mask = self._build_attn_mask(tokens.shape[-1], pad_mask)  # type: ignore
    return tokens, attn_mask

  def detokenize(self, tokens: torch.Tensor) -> List[str]:
    assert len(tokens.shape) == 1, "tokens must be shape (seqlen)"
    return self.tokenizer.decode(tokens.tolist())  # type: ignore

  def generate(
    self,
    tokens: torch.Tensor,
    attn_mask: torch.Tensor,
    sampler: Sampler.Type,
    temp: float,
    **kwargs,
  ):
    assert len(tokens.shape) == 2, "tokens must be shape (bsz, seqlen)"
    chat_id = hashlib.md5(str(time.time()).encode("utf-8")).hexdigest()
    cur_pos = 0

    while cur_pos < self.params.max_seqlen:
      if cur_pos == 0:
        x = tokens
        attn_mask = attn_mask
        freqs_cis = self.freqs_cis_all[: tokens.shape[-1]]
      else:
        x = tokens[:, -1:]
        attn_mask = torch.tensor([0]).to(device=device)
        freqs_cis = self.freqs_cis_all[cur_pos : cur_pos + 1]
      logits, self.kvcache, scores = self.xfmr(
        x, self.kvcache, cur_pos, attn_mask, freqs_cis
      )  # type: ignore
      metrics = Sampler.calculate_metrics(logits, scores)
      candidates, logprobs = Sampler.get_candidates(
        logits, sampler, temp=temp, metrics=metrics, **kwargs
      )
      next_token, _ = Sampler.random_choice(candidates, logprobs, sampler)
      is_stop = next_token[0] in self.stop_tokens
      cur_pos = tokens.shape[-1] if cur_pos == 0 else cur_pos + 1

      yield {
        "id": f"chatcmpl-{chat_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "Llama-3.2-1B-Instruct" if self.is_instruct else "Llama-3.2-1B",
        "current_token_position": cur_pos,
        "choices": [
          {
            "index": 0,
            "delta": {"content": self.tokenizer.decode(next_token[0].tolist())},  # type: ignore
            "finish_reason": "stop" if is_stop else None,
          }
        ],
        "top_predictions": [
          {"token": self.tokenizer.decode([token]), "logprob": logprob}
          for token, logprob in zip(candidates[0].tolist(), logprobs[0].tolist())  # type: ignore
        ],
        "logits_entropy": metrics["logits_entropy"].item(),
        "logits_varentropy": metrics["logits_varentropy"].item(),
        "attn_entropy": metrics["attn_entropy"].item(),
        "attn_varentropy": metrics["attn_varentropy"].item(),
      }
      if is_stop:
        break
      tokens = torch.cat([tokens, next_token], dim=1)


if __name__ == "__main__":
  is_instruct = True
  weight_path, tok_path = "src/model/1B", "src/tokenizer.model"
  weight_path = weight_path + "-Instruct" if is_instruct else weight_path
  prompts = ["What is Alzheimer's disease?"]
  llama = Llama(is_instruct, LLAMA_1B_PARAMS, weight_path, tok_path, len(prompts))
  tokens, attn_mask = llama.tokenize(prompts, format_instruct=True)
  print(llama.detokenize(tokens[0]))
  for chunk in llama.generate(tokens, attn_mask, sampler="entropix", temp=0.6, k=40):
    # print(chunk)
    print(chunk["choices"][0]["delta"]["content"], end="", flush=True)

# https://github.com/xjdr-alt/entropix/blob/main/download_weights.py
import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


def translate_key(in_key: str):
  out_key = in_key.replace(".weight", "")
  if out_key.startswith("model."):
    out_key = out_key.replace("model.", "")
    if out_key.endswith("input_layernorm"):
      out_key = out_key.replace("input_layernorm", "attention_norm")
    elif out_key.endswith("mlp.down_proj"):
      out_key = out_key.replace("mlp.down_proj", "feed_forward.w2")
    elif out_key.endswith("mlp.gate_proj"):
      out_key = out_key.replace("mlp.gate_proj", "feed_forward.w1")
    elif out_key.endswith("mlp.up_proj"):
      out_key = out_key.replace("mlp.up_proj", "feed_forward.w3")
    elif out_key.endswith("post_attention_layernorm"):
      out_key = out_key.replace("post_attention_layernorm", "ffn_norm")
    elif out_key.endswith("self_attn.k_proj"):
      out_key = out_key.replace("self_attn.k_proj", "attention.wk")
    elif out_key.endswith("self_attn.o_proj"):
      out_key = out_key.replace("self_attn.o_proj", "attention.wo")
    elif out_key.endswith("self_attn.q_proj"):
      out_key = out_key.replace("self_attn.q_proj", "attention.wq")
    elif out_key.endswith("self_attn.v_proj"):
      out_key = out_key.replace("self_attn.v_proj", "attention.wv")
    elif out_key.endswith("down_proj"):
      out_key = out_key.replace("down_proj", "w2")
    elif out_key.endswith("gate_proj"):
      out_key = out_key.replace("gate_proj", "w1")
    elif out_key.endswith("up_proj"):
      out_key = out_key.replace("up_proj", "w3")
    elif out_key == "embed_tokens":
      out_key = "tok_embeddings"
    elif out_key == "norm":
      out_key = "norm"
    else:
      raise ValueError(f"Don't know how to handle {in_key=}")
  elif out_key == "lm_head":
    out_key = "output"
  else:
    raise ValueError(f"Don't know how to handle {in_key=}")
  return f"{out_key}.weight"


def reverse_permute(
  tensor: torch.Tensor, n_heads: int = 32, dim1: int = 4096, dim2: int = 4096
) -> torch.Tensor:
  return (
    tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2)
    .transpose(1, 2)
    .reshape(dim1, dim2)
  )


if __name__ == "__main__":
  is_instruct = True
  base_model_id = "meta-llama/Llama-3.2-1B"
  model_id = f"{base_model_id}{'-Instruct' if is_instruct else ''}"
  out_dir = Path(f"src/model/1B{'-Instruct' if is_instruct else ''}")
  if not out_dir.exists():
    out_dir.mkdir(parents=True, exist_ok=True)

  token = os.environ.get("HUGGINGFACE_TOKEN")
  hf_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    offload_folder="/tmp/offload",
    token=token,
  )
  with torch.no_grad():
    state_dict = hf_model.state_dict()
    weights = {}
    for hf_name, param in state_dict.items():
      print(f" {hf_name}: {param.shape=}")
      name = translate_key(hf_name)
      param = param.clone()
      if name.endswith("wq.weight"):
        param = reverse_permute(param, n_heads=32, dim1=2048, dim2=2048)
      elif name.endswith("wk.weight"):
        param = reverse_permute(param, n_heads=8, dim1=512, dim2=2048)
      weights[name] = param
    save_file(weights, f"{out_dir}/model.safetensors")

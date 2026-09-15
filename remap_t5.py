#!/usr/bin/env python
"""Remap Wan-AI raw T5/UMT5-XXL keys to ComfyUI's expected format.
Wan-AI keys:  token_embedding.weight, blocks.N.attn.q.weight, blocks.N.ffn.fc1.weight, norm.weight
ComfyUI keys: shared.weight, encoder.block.N.layer.0.SelfAttention.q.weight,
              encoder.block.N.layer.1.DenseReluDense.wi_1.weight, encoder.final_layer_norm.weight
"""
import torch, re, os
from safetensors.torch import save_file

SRC = r"C:\Users\Administrator\comfy\ComfyUI\models\text_encoders\umt5_xxl_umt5-xxl-enc-bf16.pth"
DST = r"C:\Users\Administrator\comfy\ComfyUI\models\text_encoders\umt5_xxl_umt5-xxl-enc-bf16_comfy.safetensors"

def remap_key(k):
    if k == "token_embedding.weight":
        return "shared.weight"
    if k == "norm.weight":  # T5Encoder final norm -> T5Stack.final_layer_norm
        return "encoder.final_layer_norm.weight"
    m = re.match(r'^blocks\.(\d+)\.(.+)$', k)
    if not m:
        return None
    n, rest = m.group(1), m.group(2)
    if rest == "norm1.weight":
        return f"encoder.block.{n}.layer.0.layer_norm.weight"
    m2 = re.match(r'^attn\.(q|k|v|o)\.weight$', rest)
    if m2:
        return f"encoder.block.{n}.layer.0.SelfAttention.{m2.group(1)}.weight"
    if rest == "norm2.weight":
        return f"encoder.block.{n}.layer.1.layer_norm.weight"
    if rest == "ffn.gate.0.weight":
        return f"encoder.block.{n}.layer.1.DenseReluDense.wi_0.weight"
    if rest == "ffn.fc1.weight":
        return f"encoder.block.{n}.layer.1.DenseReluDense.wi_1.weight"
    if rest == "ffn.fc2.weight":
        return f"encoder.block.{n}.layer.1.DenseReluDense.wo.weight"
    if rest == "pos_embedding.embedding.weight":
        return f"encoder.block.{n}.layer.0.SelfAttention.relative_attention_bias.weight"
    return None

print(f"Loading {SRC} ...", flush=True)
sd = torch.load(SRC, map_location="cpu", weights_only=True)
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
print(f"Loaded {len(sd)} tensors", flush=True)

new_sd, unmapped = {}, []
for k, v in sd.items():
    nk = remap_key(k)
    if nk is None:
        unmapped.append(k)
    else:
        new_sd[nk] = v

print(f"Remapped: {len(new_sd)} tensors, unmapped: {len(unmapped)}", flush=True)
if unmapped:
    print(f"  Unmapped: {unmapped[:10]}", flush=True)
detect_key = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
print(f"Detection key present: {detect_key in new_sd}", flush=True)

save_file(new_sd, DST)
print(f"Saved {len(new_sd)} tensors -> {DST} ({os.path.getsize(DST)/1e9:.2f} GB)", flush=True)
print("DONE", flush=True)

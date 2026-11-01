import os, torch
from safetensors.torch import save_file, load_file

f = r"C:\Users\Administrator\comfy\ComfyUI\models\text_encoders\umt5_xxl_umt5-xxl-enc-bf16_comfy.safetensors"
spiece_path = r"C:\Users\Administrator\AppData\Local\Temp\wan_dl\spiece_umt5xxl.model"

sd = load_file(f)
print(f"loaded {len(sd)} tensors; has spiece_model: {'spiece_model' in sd}", flush=True)

with open(spiece_path, "rb") as fh:
    raw = fh.read()
print(f"spiece raw: {len(raw)} bytes", flush=True)

# Store as a uint8 1D tensor; SPieceTokenizer will do .numpy().tobytes()
sd["spiece_model"] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()

save_file(sd, f)
print(f"saved {len(sd)} tensors -> {f}", flush=True)
print(f"size: {os.path.getsize(f)/1e9:.2f} GB", flush=True)
print("DONE", flush=True)

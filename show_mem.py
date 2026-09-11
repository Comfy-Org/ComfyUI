import torch
import comfy.model_management as mm

GB = 1024 ** 3
device = mm.get_torch_device()

print("=" * 60)
print("GPU VRAM DIAGNOSTIC")
print("=" * 60)

print(f"Device:            {torch.cuda.get_device_name(device)}")

# Physical GPU memory as reported by the runtime/driver
free, total = torch.cuda.mem_get_info(device)

print("\n[GPU / DRIVER]")
print(f"Total VRAM:         {total / GB:.2f} GB")
print(f"Used VRAM:          {(total - free) / GB:.2f} GB")
print(f"Free VRAM:          {free / GB:.2f} GB")

# PyTorch allocator
print("\n[PYTORCH]")
print(f"Allocated:          {torch.cuda.memory_allocated(device) / GB:.2f} GB")
print(f"Reserved:           {torch.cuda.memory_reserved(device) / GB:.2f} GB")

# ComfyUI's memory manager
comfy_total = mm.get_total_memory(device)
comfy_free = mm.get_free_memory(device)

print("\n[COMFYUI]")
print(f"Total VRAM:         {comfy_total / GB:.2f} GB")
print(f"Free VRAM:          {comfy_free / GB:.2f} GB")
print(f"Used VRAM:          {(comfy_total - comfy_free) / GB:.2f} GB")

print("=" * 60)

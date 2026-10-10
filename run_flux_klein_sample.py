"""Headless runner for the One Node · FLUX.2 [klein] ComfyUI custom node.

Drives the node's bundled workflow JSON files through the ComfyUI API without
the browser UI. Auto-detects the installed model variant (4B vs 9B) and supports
T2I / I2I / EDIT modes.

Usage:
    # Text-to-image (random seed unless FK_SEED is set)
    python run_flux_klein_sample.py --prompt "a neon-noir street at night"

    # Image-to-image (variation of an existing image)
    python run_flux_klein_sample.py --mode i2i --image input.png --strength 0.6

    # Edit (describe the change to a reference image)
    python run_flux_klein_sample.py --mode edit --image input.png \
        --prompt "replace the sign with NEONFALL in cyan neon"

Env:
    FK_SEED        fixed seed (default: random)
    COMFY_URL      API base URL (default: http://127.0.0.1:8188)
    COMFY_DIR      ComfyUI install dir (default: parent of this script's dir)

Requires ComfyUI to be running:
    cd <ComfyUI> && python main.py --listen 127.0.0.1 --port 8188
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
COMFY_DIR = os.environ.get("COMFY_DIR", os.path.dirname(HERE))
NODE_DIR = os.path.join(COMFY_DIR, "custom_nodes", "one-node-flux-2-klein")
BASE = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")

NEGATIVE = "low quality, deformed, blurry, watermark, ugly, bad anatomy"
STEPS = 4
WIDTH, HEIGHT = 1024, 1024
PREFIX = "one-node-flux-2-klein/f2k"


def _detect_models():
    """Pick the installed diffusion model / text encoder / vae by scanning folders."""
    import folder_paths  # type: ignore  (only available inside ComfyUI venv)
    diffs = folder_paths.get_filename_list("diffusion_models")
    tes = folder_paths.get_filename_list("text_encoders")
    vaes = folder_paths.get_filename_list("vae")
    unet = next((d for d in diffs if "klein" in d.lower()), diffs[0] if diffs else "flux-2-klein-4b.safetensors")
    te = next((t for t in tes if "qwen" in t.lower() and "4b" in t.lower()),
              next((t for t in tes if "qwen" in t.lower()), tes[0] if tes else "qwen_3_4b_fp4_flux2.safetensors"))
    vae = next((v for v in vaes if "flux2" in v.lower() or "klein" in v.lower()),
               vaes[0] if vaes else "flux2-vae.safetensors")
    return unet, te, vae


def load_workflow(mode):
    with open(os.path.join(NODE_DIR, "workflows", f"{mode}_workflow.json"), "r", encoding="utf-8") as f:
        wf = json.load(f)
    unet, te, vae = _detect_models()
    # Swap 9B template names -> installed model files (node ships 9B defaults).
    if "FK:155" in wf:
        wf["FK:155"]["inputs"]["clip_name"] = te
    if "FK:165" in wf:
        wf["FK:165"]["inputs"]["unet_name"] = unet
    if "FK:153" in wf:
        wf["FK:153"]["inputs"]["vae_name"] = vae
    return wf


def apply_common(wf, prompt, seed):
    if "FK:166" in wf:
        wf["FK:166"]["inputs"]["text"] = prompt
    if "FK:156" in wf and "text" in wf["FK:156"]["inputs"]:
        wf["FK:156"]["inputs"]["text"] = NEGATIVE
    if "FK:170" in wf:
        wf["FK:170"]["inputs"]["width"] = WIDTH
        wf["FK:170"]["inputs"]["height"] = HEIGHT
    if "FK:171" in wf:
        wf["FK:171"]["inputs"]["seed"] = seed
        wf["FK:171"]["inputs"]["steps"] = STEPS
    if "FK:86" in wf:
        wf["FK:86"]["inputs"]["filename_prefix"] = PREFIX
    return wf


def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def get_json(path):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def upload_image(path):
    """Upload an input image via the ComfyUI /upload endpoint; returns the filename."""
    boundary = "----fluxkleinboundary"
    with open(path, "rb") as f:
        raw = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{os.path.basename(path)}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + raw + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        BASE + "/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))["name"]


def build_prompt(mode, args, seed):
    wf = load_workflow(mode)
    wf = apply_common(wf, args.prompt, seed)
    if mode in ("i2i", "edit"):
        fname = upload_image(args.image)
        # Reference image node differs per workflow; set the first LoadImage input.
        img_node = next((k for k, v in wf.items() if v.get("class_type") == "LoadImage"), None)
        if img_node:
            wf[img_node]["inputs"]["image"] = fname
        if mode == "i2i" and "FK:171" in wf:
            wf["FK:171"]["inputs"]["denoise"] = args.strength
    return wf


def main():
    ap = argparse.ArgumentParser(description="Headless FLUX.2 [klein] runner")
    ap.add_argument("--mode", choices=["t2i", "i2i", "edit"], default="t2i")
    ap.add_argument("--prompt", default="a neon-noir cyberpunk street at night, rain-slicked asphalt reflecting "
                     "holographic signage, a lone figure in a glowing trench coat, cinematic lighting, volumetric "
                     "fog, ultra detailed, 8k")
    ap.add_argument("--image", default=None, help="input image for i2i/edit modes")
    ap.add_argument("--strength", type=float, default=0.6, help="i2i denoise strength")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("FK_SEED", random.randint(0, 2**32 - 1))))
    args = ap.parse_args()

    if args.mode in ("i2i", "edit") and not args.image:
        print("ERROR: --image is required for i2i/edit modes", file=sys.stderr)
        sys.exit(1)

    wf = build_prompt(args.mode, args, args.seed)
    resp = post_json("/prompt", {"prompt": wf, "client_id": "flux_klein_sample"})
    if "error" in resp:
        print("QUEUE ERROR:", json.dumps(resp, indent=2))
        sys.exit(1)
    prompt_id = resp["prompt_id"]
    print(f"Queued {args.mode} prompt_id={prompt_id} seed={args.seed}")

    out_dir = os.path.join(COMFY_DIR, "output", "one-node-flux-2-klein")
    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            hist = get_json(f"/history/{prompt_id}")
        except Exception:
            hist = {}
        if prompt_id in hist:
            imgs = [im["filename"] for node in hist[prompt_id].get("outputs", {}).values()
                    for im in node.get("images", [])]
            print("DONE. outputs:", imgs)
            if os.path.isdir(out_dir):
                files = sorted(
                    [f for f in os.listdir(out_dir) if f.startswith("f2k") and f.endswith(".png")],
                    key=lambda f: os.path.getmtime(os.path.join(out_dir, f)), reverse=True,
                )
                if files:
                    print("LATEST:", os.path.join(out_dir, files[0]))
            return
        time.sleep(3)
    print("TIMEOUT waiting for completion")
    sys.exit(2)


if __name__ == "__main__":
    main()

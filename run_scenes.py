#!/usr/bin/env python
# Wan 2.1 1.3B T2V scene driver. Waits for models, ensures ComfyUI is up (pinned to
# GPU1 / A4000, low-VRAM), generates each scene via REST, ffmpeg-encodes to MP4.
# Robust + idempotent: skips scenes whose MP4 already exists. Logs to run.log.
import json, os, sys, time, shutil, glob, subprocess, urllib.request, urllib.error

ROOT = r"C:\Users\Administrator\comfy\ComfyUI"
SK   = r"C:\Users\Administrator\AppData\Local\hermes\profiles\short-video-director\skills\creative\comfyui\scripts"
BASE = "http://127.0.0.1:8188"
WORKFLOW = os.path.join(ROOT, "wan_t2v_scene.json")
SCENES   = os.path.join(ROOT, "scenes.json")
OUTDIR   = os.path.join(ROOT, "video_out")
LOG      = os.path.join(ROOT, "run.log")
os.makedirs(OUTDIR, exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def http(url, data=None, headers=None, timeout=30):
    h = {"Content-Type": "application/json"}
    if headers: h.update(headers)
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")

def server_up():
    try:
        http(BASE + "/system_stats", timeout=5)
        return True
    except Exception:
        return False

def wait_models(max_wait=4*3600):
    need = {
        "base": (os.path.join(ROOT, "models/diffusion_models/wan2.1_t2v_1.3B.safetensors"), 5676070424),
        "t5":   (os.path.join(ROOT, "models/text_encoders/umt5_xxl_umt5-xxl-enc-bf16.pth"),   11361920418),
        "vae":  (os.path.join(ROOT, "models/vae/wan_2.1_vae.pth"),                             507609880),
    }
    t0 = time.time()
    while True:
        ok = True
        for k, (p, sz) in need.items():
            if not os.path.exists(p) or os.path.getsize(p) < int(sz * 0.999):
                ok = False
                if os.path.exists(p):
                    log(f"  {k}: {os.path.getsize(p):,} / {sz:,} bytes ({100*os.path.getsize(p)/sz:.1f}%)")
                else:
                    log(f"  {k}: not present yet")
        if ok:
            log("All model files present & full-size.")
            return True
        if time.time() - t0 > max_wait:
            log("TIMED OUT waiting for models")
            return False
        time.sleep(30)

def launch_comfyui():
    if server_up():
        log("ComfyUI already up.")
        return
    log("Launching ComfyUI server (pinned to GPU1/A4000, low-VRAM)...")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    exe = os.path.join(ROOT, ".venv/Scripts/python.exe")
    args = [exe, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--lowvram",
            "--dont-upcast-attention", "--reserve-vram", "2.5"]
    logf = open(os.path.join(ROOT, "comfyui_server.log"), "a", encoding="utf-8")
    subprocess.Popen(args, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
    # wait up to 120s for health
    t0 = time.time()
    while time.time() - t0 < 180:
        if server_up():
            log("ComfyUI is up.")
            return
        time.sleep(3)
    log("WARNING: ComfyUI not up after 180s; continuing and will retry per-scene.")

def build_workflow(prompt, seed):
    wf = json.load(open(WORKFLOW, encoding="utf-8"))
    wf["6"]["inputs"]["text"] = prompt
    wf["3"]["inputs"]["seed"] = int(seed)
    return wf

def run_scene(scene, seed):
    sid = scene["id"]
    final = os.path.join(OUTDIR, f"{sid}.mp4")
    if os.path.exists(final) and os.path.getsize(final) > 20000:
        log(f"SKIP {sid}: MP4 already exists")
        return {"id": sid, "mp4": final, "status": "cached"}
    wf = build_workflow(scene["prompt"], seed)
    log(f"SUBMIT {sid}")
    r = http(BASE + "/prompt", {"prompt": wf, "client_id": "hermes-" + sid})
    pid = r.get("prompt_id")
    if not pid:
        raise RuntimeError(f"no prompt_id: {r}")
    # poll history
    t0 = time.time()
    while time.time() - t0 < 3600:
        try:
            h = http(BASE + "/history/" + pid, timeout=15)
        except Exception:
            time.sleep(5); continue
        if pid in h and h[pid].get("outputs"):
            out = h[pid]["outputs"]
            status = h[pid].get("status", {}).get("status_str", "unknown")
            # find image outputs
            files = []
            for nodeout in out.values():
                for im in nodeout.get("images", []):
                    files.append(im["filename"])
            if status == "success" and files:
                log(f"  {sid}: done in {time.time()-t0:.0f}s, {len(files)} frames")
                # copy frames to a scene folder, in numeric order
                sdir = os.path.join(OUTDIR, sid + "_frames")
                os.makedirs(sdir, exist_ok=True)
                def frame_idx(fn):
                    import re
                    m = re.search(r"(\d+)\.", fn)
                    return int(m.group(1)) if m else 0
                files = sorted(files, key=frame_idx)
                for i, fn in enumerate(files):
                    src = os.path.join(ROOT, "output", fn)
                    dst = os.path.join(sdir, f"f{i:04d}.png")
                    if os.path.exists(src):
                        shutil.copy(src, dst)
                nframes = len(glob.glob(os.path.join(sdir, "f*.png")))
                # ffmpeg encode
                ff = shutil.which("ffmpeg") or r"C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
                cmd = [ff, "-y", "-framerate", "16", "-i", os.path.join(sdir, "f%04d.png"),
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                       "-movflags", "+faststart", final]
                log(f"  {sid}: ffmpeg encoding {nframes} frames -> {os.path.basename(final)}")
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if p.returncode != 0:
                    log("  ffmpeg stderr: " + (p.stderr[-800:] if p.stderr else "?"))
                    raise RuntimeError("ffmpeg failed")
                sz = os.path.getsize(final)
                log(f"  {sid}: MP4 ready {sz/1e6:.1f} MB")
                return {"id": sid, "mp4": final, "status": "ok", "frames": nframes, "bytes": sz}
            elif status == "error":
                raise RuntimeError(f"ComfyUI error for {sid}: " + json.dumps(out)[:800])
        time.sleep(5)
    raise RuntimeError(f"timeout waiting for {sid}")

def main():
    log("=== Wan scene driver start ===")
    log(f"GPU pin: CUDA_VISIBLE_DEVICES=1 (A4000)")
    if not wait_models():
        log("FATAL: models never ready"); sys.exit(2)
    launch_comfyui()
    # small settle for first model load
    time.sleep(5)
    scenes = json.load(open(SCENES, encoding="utf-8"))["scenes"]
    results = []
    # Order: scene 1 first (highest priority = "at least one"), then the rest
    order = [scenes[0]] + scenes[1:]
    for i, sc in enumerate(order):
        seed = 20260828 + i
        for attempt in range(2):
            try:
                res = run_scene(sc, seed)
                results.append(res)
                break
            except Exception as e:
                log(f"  {sc['id']} attempt {attempt+1} FAILED: {e}")
                if attempt == 0:
                    log("  freeing VRAM + retrying with a fresh seed")
                    try:
                        http(BASE + "/interrupt", timeout=10)
                    except Exception:
                        pass
                    time.sleep(8)
                else:
                    results.append({"id": sc["id"], "status": "failed", "error": str(e)})
        # checkpoint manifest after each
    manifest = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "Wan2.1-T2V-1.3B", "resolution": "832x480", "fps": 16,
                "results": results}
    with open(os.path.join(OUTDIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    okc = sum(1 for r in results if r["status"] in ("ok", "cached"))
    log(f"=== DONE: {okc}/{len(results)} scenes produced MP4 ===")
    for r in results:
        log(f"   {r['id']}: {r['status']}" + (f" ({r.get('bytes',0)/1e6:.1f} MB)" if r.get('bytes') else ""))
    sys.exit(0 if okc >= 1 else 3)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Batch scenes 2-13 on ComfyUI port 8189. Sequential, encode each to MP4."""
import json, time, os, shutil, glob, subprocess, urllib.request, re

BASE = "http://127.0.0.1:8189"
ROOT = r"C:\Users\Administrator\comfy\ComfyUI"
OUT = os.path.join(ROOT, "video_out")
os.makedirs(OUT, exist_ok=True)
FFMPEG = r"C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"

scenes = json.load(open(os.path.join(ROOT, "scenes.json"), encoding="utf-8"))["scenes"]
wf_template = json.load(open(os.path.join(ROOT, "wan_t2v_scene.json"), encoding="utf-8"))

def http(url, data=None, timeout=30):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")

def render_scene(sc, idx, seed):
    sid = sc["id"]
    final_mp4 = os.path.join(OUT, f"{sid}.mp4")
    if os.path.exists(final_mp4) and os.path.getsize(final_mp4) > 10000:
        print(f"  SKIP {sid} (already exists)", flush=True)
        return True

    wf = json.loads(json.dumps(wf_template))  # deep copy
    wf["6"]["inputs"]["text"] = sc["prompt"]
    wf["3"]["inputs"]["seed"] = seed
    wf["10"]["inputs"]["filename_prefix"] = f"scene_{idx:02d}"

    print(f"  SUBMIT {sid} (seed={seed})...", flush=True)
    r = http(BASE + "/prompt", {"prompt": wf, "client_id": f"hermes-{sid}"})
    pid = r.get("prompt_id")
    if not pid:
        print(f"  FAIL: no prompt_id: {r}", flush=True)
        return False

    t0 = time.time()
    while time.time() - t0 < 3600:
        try:
            h = http(BASE + "/history/" + pid, timeout=15)
        except Exception:
            time.sleep(5)
            continue
        if pid in h:
            entry = h[pid]
            status = entry.get("status", {}).get("status_str", "?")
            files = [im["filename"] for no in entry.get("outputs", {}).values() for im in no.get("images", [])]
            if status == "success" and files:
                elapsed = time.time() - t0
                print(f"  RENDER OK {sid} in {elapsed:.0f}s, {len(files)} frames", flush=True)
                # filename_prefix = f"scene_{idx:02d}" -> files are <prefix>_00001_.png ...
                pattern = os.path.join(ROOT, "output", f"scene_{idx:02d}_%05d_.png")
                cmd = [FFMPEG, "-y", "-framerate", "16", "-start_number", "1",
                       "-i", pattern,
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                       "-movflags", "+faststart", final_mp4]
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if p.returncode != 0:
                    print(f"  FFMPEG FAIL {sid}: {p.stderr[-300:]}", flush=True)
                    return False
                sz = os.path.getsize(final_mp4)
                print(f"  MP4 OK {sid}: {sz/1e6:.1f} MB", flush=True)
                return True
            elif status == "error":
                msgs = entry.get("status", {}).get("messages", [])
                err_msg = ""
                for m in msgs:
                    if m[0] == "execution_error":
                        err_msg = m[1].get("exception_message", "")[:300]
                print(f"  RENDER ERR {sid}: {err_msg}", flush=True)
                return False
        time.sleep(8)
    print(f"  TIMEOUT {sid}", flush=True)
    return False

results = []
for idx, sc in enumerate(scenes):
    sid = sc["id"]
    final_mp4 = os.path.join(OUT, f"{sid}.mp4")
    if os.path.exists(final_mp4) and os.path.getsize(final_mp4) > 10000:
        print(f"[{idx+1}/{len(scenes)}] SKIP {sid}", flush=True)
        results.append({"id": sid, "status": "cached"})
        continue
    seed = 20260828 + idx
    print(f"[{idx+1}/{len(scenes)}] {sc['label']}", flush=True)
    ok = render_scene(sc, idx + 1, seed)
    results.append({"id": sid, "status": "ok" if ok else "failed"})
    # Small pause between scenes to let VRAM settle
    if ok:
        time.sleep(3)

manifest = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model": "Wan2.1-T2V-1.3B",
    "resolution": "832x480",
    "fps": 16,
    "results": results
}
with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

ok_count = sum(1 for r in results if r["status"] in ("ok", "cached"))
print(f"\n=== DONE: {ok_count}/{len(results)} scenes ===", flush=True)
for r in results:
    mp4 = os.path.join(OUT, r["id"] + ".mp4")
    sz = os.path.getsize(mp4) if os.path.exists(mp4) else 0
    print(f"  {r['id']}: {r['status']} ({sz/1e6:.1f} MB)", flush=True)

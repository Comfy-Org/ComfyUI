#!/usr/bin/env python
import json, time, os, glob, subprocess, urllib.request, urllib.error, sys

BASE = "http://127.0.0.1:8189"
ROOT = r"C:\Users\Administrator\comfy\ComfyUI"
FFMPEG = r"C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
OUTDIR = os.path.join(ROOT, "video_out")

def http(url, data=None, timeout=10):
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def wait_prompt(pid, timeout=1200):
    start=time.time()
    while time.time()-start < timeout:
        try:
            h=http(BASE+f"/history/{pid}", timeout=10)
            if pid in h:
                hist=h[pid]
                status=hist.get("status",{})
                if status.get("status_str")=="error":
                    return False, hist
                if "outputs" in hist:
                    return True, hist
            q=http(BASE+"/queue", timeout=10)
            running=[x[1] for x in q.get("queue_running",[])]
            pending=[x[1] for x in q.get("queue_pending",[])]
            if pid not in running and pid not in pending:
                # check history again
                h=http(BASE+f"/history/{pid}", timeout=10)
                if pid in h and "outputs" in h[pid]:
                    return True, h[pid]
                if pid in h and h[pid].get("status",{}).get("status_str")=="error":
                    return False, h[pid]
        except Exception as e:
            print(f"  poll err: {e}")
        time.sleep(5)
    return False, {"error":"timeout"}

def run_one(wf_path, label, mp4_name):
    wf=json.load(open(wf_path, encoding="utf-8"))
    # clear old frames for this prefix
    prefix=wf["10"]["inputs"]["filename_prefix"]
    for f in glob.glob(os.path.join(ROOT,"output", prefix+"*")):
        try: os.remove(f)
        except: pass
    print(f"\n=== {label} ===", flush=True)
    print(f"  workflow: {os.path.basename(wf_path)}", flush=True)
    r=http(BASE+"/prompt", {"prompt":wf, "client_id":f"nail-{label}"})
    pid=r.get("prompt_id")
    if not pid:
        print(f"  FAILED submit: {r}")
        return False
    print(f"  SUBMIT {pid} ...", flush=True)
    ok, hist=wait_prompt(pid, timeout=1200)
    if not ok:
        print(f"  FAILED render: {hist.get('status',hist) if isinstance(hist,dict) else hist}", flush=True)
        return False
    # find frames
    outs=hist.get("outputs",{})
    frames=[]
    for nid, node_out in outs.items():
        for img in node_out.get("images",[]):
            frames.append(img.get("filename"))
    if not frames:
        # fallback glob
        frames=sorted(glob.glob(os.path.join(ROOT,"output", prefix+"*.png")))
        print(f"  fallback glob found {len(frames)} pngs", flush=True)
    else:
        print(f"  got {len(frames)} frames from history", flush=True)
    # glob actual files for ffmpeg
    pngs=sorted(glob.glob(os.path.join(ROOT,"output", prefix+"*.png")))
    print(f"  pngs on disk: {len(pngs)} first={os.path.basename(pngs[0]) if pngs else 'none'}", flush=True)
    if not pngs:
        print("  no pngs!", flush=True)
        return False
    # encode
    # need pattern: prefix_%05d_.png  - find number width
    mp4=os.path.join(OUTDIR, mp4_name)
    os.makedirs(OUTDIR, exist_ok=True)
    # use ffmpeg with pattern
    pattern=os.path.join(ROOT,"output", prefix+"_%05d_.png")
    # detect actual start_number
    cmd=[FFMPEG,"-y","-framerate","16","-start_number","1","-i",pattern,"-c:v","libx264","-pix_fmt","yuv420p","-crf","18","-preset","medium","-movflags","+faststart",mp4]
    print(f"  ffmpeg -> {mp4}", flush=True)
    res=subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if os.path.exists(mp4):
        sz=os.path.getsize(mp4)
        print(f"  MP4 OK {mp4_name}: {round(sz/1024,1)} KB", flush=True)
        return True
    else:
        print(f"  ffmpeg failed: {res.stderr[-800:]}", flush=True)
        return False

# Run test 1: 1.3B 81 frames
t0=time.time()
ok1=run_one(os.path.join(ROOT,"wan_i2v_nail_81_1.3B.json"), "81frames_1.3B_I2V", "nail_81f_1.3B_I2V.mp4")
t1=time.time()
print(f"\n--- Test1 done in {round((t1-t0)/60,1)} min, ok={ok1} ---", flush=True)

# Run test 2: 14B 49 frames
t2=time.time()
ok2=run_one(os.path.join(ROOT,"wan_i2v_nail_49_14B.json"), "49frames_14B_I2V", "nail_49f_14B_I2V.mp4")
t3=time.time()
print(f"\n--- Test2 done in {round((t3-t2)/60,1)} min, ok={ok2} ---", flush=True)
print(f"\n=== BOTH DONE total {round((t3-t0)/60,1)} min ===", flush=True)
print(f"  1.3B 81f: {'OK' if ok1 else 'FAIL'}", flush=True)
print(f"  14B 49f: {'OK' if ok2 else 'FAIL'}", flush=True)

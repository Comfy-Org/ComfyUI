"""Describe a pack that may not import as it stands here.

    python tolerant.py <pack dir> <describe.json> <plan.json>

describe.py runs in a fresh process with the plan so far (empty at first). When the pack doesn't load, that
process finds what stands in the way, one failure at a time (standin.py), and writes the plan; the pack is then
described again, in another fresh process, with that plan. The describe.json kept is always from a process in
which the pack loaded at the first try: nothing in it comes from a process that imported the pack twice.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESS_SECONDS = float(os.environ.get("CONVERTER_DESCRIBE_SECONDS", "240"))
PROCESSES = 7


def _read(path):
    try:
        with open(path) as file:
            return json.load(file)
    except (OSError, ValueError):
        return None


def main(pack_dir, out_path, plan_path):
    if not os.path.exists(plan_path):
        with open(plan_path, "w") as file:
            json.dump({}, file)
    history, result, loads_with = [], None, None
    for process in range(1, PROCESSES + 1):
        plan_before = _read(plan_path)
        try:
            os.remove(out_path)
        except FileNotFoundError:
            pass
        try:
            done = subprocess.run([sys.executable, os.path.join(HERE, "describe.py"), pack_dir, out_path, plan_path],
                                  capture_output=True, text=True, errors="replace", timeout=PROCESS_SECONDS)
            code, log = done.returncode, (done.stdout + done.stderr)[-1500:]
        except subprocess.TimeoutExpired as late:
            code, log = -1, f"no result in {PROCESS_SECONDS:.0f}s: " + str((late.stderr or b"")[-600:])
        result = _read(out_path)
        plan_after = _read(plan_path)
        if result is None:  # the process died (a crash, an exit, a hang) before it could say anything
            result = {"pack": pack_dir, "ok": False, "error": f"describe exited {code}: {log}"[:2000], "stand_ins": plan_after}
        history.append({"process": process, "ok": bool(result.get("ok")), "error": (result.get("error") or "")[:300],
                        "attempts": result.get("attempts"), "found": result.get("found")})
        if result.get("loads_with") is not None:
            loads_with = result["loads_with"]  # it loads with this plan; more of its nodes may load with the new one
        if result.get("ok") or plan_after == plan_before:
            break  # described, or nothing more was found to stand in
    if not result.get("ok") and loads_with is not None:  # looking for more left it not loading: as it loaded before
        with open(plan_path, "w") as file:
            json.dump(dict(loads_with, leave_caught=True), file)
        try:
            os.remove(out_path)
        except FileNotFoundError:
            pass
        try:
            subprocess.run([sys.executable, os.path.join(HERE, "describe.py"), pack_dir, out_path, plan_path],
                           capture_output=True, text=True, errors="replace", timeout=PROCESS_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        again = _read(out_path)
        if again is not None:
            result = again
            history.append({"process": len(history) + 1, "ok": bool(result.get("ok")), "error": (result.get("error") or "")[:300],
                            "as_before": True})
    result["processes"] = history
    with open(out_path, "w") as file:
        json.dump(result, file)


if __name__ == "__main__":
    main(os.path.realpath(sys.argv[1]), sys.argv[2], sys.argv[3])

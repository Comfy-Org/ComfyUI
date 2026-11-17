"""Convert one V1 custom-node pack to the V4 API, in one command:

    python comfy_api/conversion/convert_pack.py <pack dir> [--out <dir>]

Each step that imports the pack runs in a fresh CPU ComfyUI process:
1. describe.py: imports the pack as pack isolation would, and records what ComfyUI lists for each node;
2. convert.py: writes <out>/<pack>/, the pack's files with known engine uses rewritten to the API, and
   v2/ (_v2_nodes.py, _v2_engine.py, _v2_routes.py, pyproject.toml, __init__.py);
3. verify.py: checks the converted pack against the original: declarations, schemas, and each node's
   outputs on made-up inputs;
4. review.py: notes what a person or an agent must still convert or test, as `# V2-MANUAL` / `# V2-TEST`
   comments on the lines and v2/V2_CONVERSION.md (comfy_api/conversion/SKILL.md is how an agent finishes).
The result is one of: verified, needs-testing, needs-manual, none; the details are in
<out>/<pack>.work/conversion.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMFY_ROOT = os.path.dirname(os.path.dirname(HERE))


def child(script, *args, timeout):
    env = {**os.environ, "COMFY_ROOT": COMFY_ROOT}
    done = subprocess.run([sys.executable, os.path.join(HERE, script), *args], cwd=COMFY_ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)
    return done.returncode, (done.stdout + done.stderr)[-2000:]


def load(path, fallback):
    try:
        with open(path) as file:
            return json.load(file)
    except (OSError, ValueError):
        return fallback


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pack", help="the V1 pack's directory")
    parser.add_argument("--out", default="converted", help="where <pack>/ is written (default: ./converted)")
    args = parser.parse_args(argv)
    pack_dir = os.path.realpath(args.pack)
    name = os.path.basename(os.path.normpath(pack_dir))
    out = os.path.realpath(args.out)
    work = os.path.join(out, name + ".work")
    os.makedirs(work, exist_ok=True)

    core = os.environ.get("CONVERTER_CORE_NODES") or os.path.join(out, "core_nodes.json")
    if not os.path.exists(core):
        print("reading ComfyUI's own nodes, as the converter calls them...")  # noqa: T201
        code, log = child("core_nodes.py", core, timeout=900)
        if code:
            sys.exit(f"core_nodes.py failed:\n{log}")
    os.environ["CONVERTER_CORE_NODES"] = core
    sys.path.insert(0, HERE)
    import convert  # noqa: E402  reads the table at import
    import review  # noqa: E402

    describe_path = os.path.join(work, "describe.json")
    code, log = child("describe.py", pack_dir, describe_path, timeout=900)
    describe = load(describe_path, {"ok": False, "error": f"describe exited {code}: {log}"})
    record = {"pack": name, "dir": pack_dir, "describe": {k: v for k, v in describe.items() if k != "nodes"},
              "node_kinds": {k: v.get("kind") for k, v in (describe.get("nodes") or {}).items()}}
    if not describe.get("ok") or not any(e.get("kind") in ("v1", "v3") for e in (describe.get("nodes") or {}).values()):
        record.update({"class": "none", "why": describe.get("error") or "no nodes to convert"})
        return finish(record, work)

    record["conversion"] = convert.generate(pack_dir, describe, out)
    converted = os.path.join(out, name)
    v2 = os.path.join(converted, "v2")
    verify_path = os.path.join(work, "verify.json")
    code, log = child("verify.py", pack_dir, converted, describe_path, verify_path, timeout=3600)
    verify = load(verify_path, {"valid": False, "error": f"verify exited {code}: {log}"})
    record["verify"] = verify
    if not verify.get("valid"):
        record.update({"class": "none", "why": verify.get("error") or "the converted pack isn't valid"})
        return finish(record, work)

    conversion = record["conversion"]
    flags = review.review_files(v2, conversion) + review.review_routes(conversion, verify) \
        + review.review_frontend(pack_dir, conversion.get("namespace")) + review.review_nodes(v2, describe, verify)
    review.review_runtime(verify, flags)
    record["class"] = review.classify(record, flags)
    record["flags"] = flags
    review.mark_lines(v2, flags)
    with open(os.path.join(v2, "V2_CONVERSION.md"), "w", encoding="utf-8") as file:
        file.write(review.worklist(name, {"class": record["class"], "flags": flags}))
    for root, dirs, _ in os.walk(converted):  # verifying imported the pack
        for cache in [d for d in dirs if d == "__pycache__"]:
            shutil.rmtree(os.path.join(root, cache), ignore_errors=True)
    return finish(record, work)


def finish(record, work) -> int:
    with open(os.path.join(work, "conversion.json"), "w") as file:
        json.dump(record, file, indent=1)
    flags = record.get("flags") or []
    manual = sum(f["kind"] == "manual" for f in flags)
    test = sum(f["kind"] == "test" for f in flags)
    print(f"{record['pack']}: {record['class']}" + (f" ({record['why']})" if record.get("why") else "")  # noqa: T201
          + (f": {manual} to convert by hand, {test} to test; see v2/V2_CONVERSION.md" if flags else ""))
    print(f"details: {os.path.join(work, 'conversion.json')}")  # noqa: T201
    return 0 if record["class"] != "none" else 1


if __name__ == "__main__":
    sys.exit(main())

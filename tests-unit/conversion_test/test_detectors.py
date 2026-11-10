"""The review detectors: each flags what it should, and only that."""
import os as _os
import sys as _sys
_CONVERSION = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion")
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault("COMFY_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_os.environ["PYTHONPATH"] = _os.pathsep.join(p for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", "")) if p)
import ast
import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import review  # noqa: E402


def writes(code):
    return [what for _line, what in review.file_writes(ast.parse(textwrap.dedent(code)))]


@pytest.mark.parametrize("code", [
    "open(path, 'w').write(x)",
    "open(path, mode='a')",
    "open(path, 'wb')",
    "open(path, 'r+')",
    "import json\nwith open(p, 'w') as f:\n    json.dump(d, f)",
    "import os\nos.makedirs(d, exist_ok=True)",
    "import os\nos.remove(p)",
    "import os as o\no.unlink(p)",
    "from os import makedirs\nmakedirs(d)",
    "import shutil\nshutil.copy(a, b)",
    "import torch\ntorch.save(sd, path)",
    "import numpy as np\nnp.save(path, arr)",
    "import cv2\ncv2.imwrite(path, img)",
    "import soundfile as sf\nsf.write(path, wav, 44100)",
    "img.save(path, format='PNG')",
    "pathlib.Path(p).write_text('x')",
    "Path(p).mkdir(parents=True)",
    "model.save_pretrained(folder)",
    "from comfy_api.v4 import util\nutil.save_torch_file(sd, path)",
    "import urllib.request\nurllib.request.urlretrieve(url, path)",
])
def test_file_writes_are_found(code):
    assert writes(code), code


@pytest.mark.parametrize("code", [
    "open(path)",
    "open(path, 'r')",
    "open(path, 'rb')",
    "with open(path, encoding='utf-8') as f:\n    data = f.read()",
    "import json\njson.load(open(p))",
    "from PIL import Image\nImage.open(path)",
    "import torch\ntorch.load(path)",
    "import io\nbuf = io.BytesIO()\nimg.save(buf, format='PNG')",
    "from io import BytesIO\nwith BytesIO() as out:\n    img.save(out, 'PNG')",
    "img.save(io.BytesIO(), 'PNG')",
    "import io, torch\nb = io.BytesIO()\ntorch.save(sd, b)",
    "import os\nos.path.exists(p)",
    "import os\nos.listdir(d)",
    "import json\ntext = json.dumps(d)",
    "settings.save({'a': 1})",
])
def test_reads_and_memory_are_not_writes(code):
    assert not writes(code), code


def _review(code, conversion=None, tmp_path=None):
    v2 = tmp_path / "v2"
    v2.mkdir(exist_ok=True)
    (v2 / "nodes.py").write_text(textwrap.dedent(code))
    return review.review_files(str(v2), conversion or {"rewritten": {"nodes.py": ["x"]}})


def test_import_time_defaults_are_flagged_only_outside_execute(tmp_path):
    flags = _review('''
        from . import _v2_engine as _v2e
        DEVICE = _v2e.get_torch_device()

        class Node:
            def run(self, x):
                return x.to(_v2e.get_torch_device())
        ''', tmp_path=tmp_path)
    lines = [f["line"] for f in flags if f["topic"] == "engine call at import"]
    assert lines == [3]


def test_a_v1_nodes_input_types_isnt_import_time(tmp_path):
    """The converted pack's schema is generated; INPUT_TYPES isn't called."""
    flags = _review('''
        from . import _v2_engine as _v2e
        class Node:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"d": (str(_v2e.get_torch_device()),)}}
        ''', tmp_path=tmp_path)
    assert not [f for f in flags if f["topic"] == "engine call at import"]


def test_real_folder_lists_arent_flagged(tmp_path):
    flags = _review('''
        from comfy_api.v4 import util
        NAMES = util.folders.get_filename_list("loras")
        ''', tmp_path=tmp_path)
    assert not [f for f in flags if f["topic"] == "engine call at import"]


def test_engine_uses_left_are_located(tmp_path):
    flags = _review('''
        import comfy.model_management

        def f(m: "comfy.model_patcher.ModelPatcher"):
            return comfy.model_management.should_use_fp16()
        ''', {"leftovers": {"nodes.py": ["comfy.model_management.should_use_fp16"]}}, tmp_path)
    assert [(f["line"], f["topic"]) for f in flags] == [(5, "engine use left")]


def test_markers_never_change_the_program(tmp_path):
    v2 = tmp_path / "v2"
    v2.mkdir()
    source = 'x = """open(p, "w")\n"""\nopen(p, "w").write(x)\ny = 1 + \\\n    2\n'
    (v2 / "a.py").write_text(source)
    flags = [{"file": "a.py", "line": n, "kind": "manual", "topic": "file write", "note": "n"} for n in (1, 3, 4)]
    review.mark_lines(str(v2), flags)
    marked = (v2 / "a.py").read_text()
    assert ast.dump(ast.parse(marked)) == ast.dump(ast.parse(source))
    assert "V2-MANUAL" in marked.split("\n")[2]  # the real write is marked
    assert "V2-MANUAL" not in marked.split("\n")[0]  # inside a string: not marked
    assert "V2-MANUAL" not in marked.split("\n")[3]  # a continued line: not marked


import convert  # noqa: E402


def _rewrite(code, typed=None):
    new, done, left = convert.rewrite(textwrap.dedent(code), "nodes.py", typed or {})
    return new.split(convert._V2_IMPORT_MARK)[-1] if done else new, done, left


def test_local_names_shadow_engine_imports():
    # KJNodes: ``def cfg_zerostar(args): args["cond"]`` became ``_v2e.args["cond"]`` (cli_args' args)
    new, _done, _left = _rewrite('''
        from comfy.cli_args import args
        import comfy.model_management as mm
        def top():
            return args.disable_metadata
        def run(model):
            def cb(args):
                return args["cond"], mm.get_torch_device()
            squares = [args * args for args in range(3)]
            def g():
                global args
                return args.preview_size
            return cb, (lambda args: args["x"]), squares
        ''')
    assert 'args["cond"]' in new and 'args["x"]' in new and "args * args" in new
    assert "_v2e.args[" not in new
    assert "_v2e.args.disable_metadata" in new and "_v2e.args.preview_size" in new
    assert "_v2e.get_torch_device()" in new


def test_changing_cli_args_is_left_for_hand_conversion():
    _new, _done, left = _rewrite('''
        from comfy.cli_args import args
        def run(on):
            if on:
                args.fast.add("cublas_ops")
            args.preview_size = 512
            return "fp16_accumulation" in args.fast
        ''')
    assert "comfy.cli_args.args.fast (changed)" in left
    assert "comfy.cli_args.args.preview_size (changed)" in left


def test_engine_object_attributes_and_setting_changes_are_located(tmp_path):
    flags = _review('''
        from . import _v2_engine as _v2e

        def execute(model, on):
            opts = model.model_options
            if on:
                _v2e.args.fast.add("cublas_ops")
            return opts
        ''', {"leftovers": {"nodes.py": ["MODEL.model_options", "comfy.cli_args.args.fast (changed)"]}}, tmp_path)
    got = [(f["line"], f["topic"]) for f in flags if f["kind"] == "manual"]
    assert got == [(5, "engine object attribute"), (7, "changes a ComfyUI setting")]


def test_v3_nodes_inputs_are_typed(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "__init__.py").write_text(textwrap.dedent('''
        from comfy_api.latest import io
        class Skim(io.ComfyNode):
            @classmethod
            def execute(cls, model, vae, pixels):
                m = model.clone()
                return m, vae.encode(pixels), model.get_model_object("model_sampling")
        NODE_CLASS_MAPPINGS = {"Skim": Skim}
        '''))
    describe = {"ok": True, "nodes": {"Skim": {"class": "Skim", "kind": "v3", "info": {"input": {
        "required": {"model": ["MODEL", {}], "vae": ["VAE", {}], "pixels": ["IMAGE", {}]}}}}}}
    conversion = convert.generate(str(pack), describe, str(tmp_path / "out"))
    code = open(tmp_path / "out" / "pack" / "v2" / "__init__.py").read()
    assert "eng.model.clone(model)" in code and "eng.vae.encode(vae, pixels)" in code
    assert "MODEL.get_model_object" in {n for v in conversion["leftovers"].values() for n in v}



def test_core_node_calls_bind_in_the_methods_order():
    # Impact's IterativeImageUpscale: nodes.VAEEncode().encode(vae, pixels); VAEEncode's schema is (pixels, vae)
    new, _done, _left = _rewrite('''
        import nodes
        def run(vae, pixels, model, clip):
            return nodes.VAEEncode().encode(vae, pixels)[0], nodes.LoraLoader().load_lora(model, clip, "x", 1.0, 1.0)
        ''')
    assert "_v2e.CoreNode('VAEEncode', ['pixels', 'vae'], 'encode', ['vae', 'pixels'])" in new
    # where the orders agree, the call is as before
    assert "_v2e.CoreNode('LoraLoader', ['model', 'clip', 'lora_name', 'strength_model', 'strength_clip'], " \
           "'load_lora')" in new


def test_core_node_stand_in_names_positional_arguments_by_the_method():
    source = convert.ENGINE_SHIM
    start = source.index("class CoreNode:")
    end = source.index("\n\n\n", start)
    calls = []
    namespace = {"sync": lambda result: result,
                 "eng": type("E", (), {"node": type("N", (), {"run": staticmethod(lambda i, a: calls.append((i, a)))})})}
    exec(source[start:end], namespace)  # noqa: S102
    core = namespace["CoreNode"]
    core("VAEEncode", ["pixels", "vae"], "encode", ["vae", "pixels"])().encode("V", "P")
    core("VAEDecode", ["samples", "vae"], "decode")().decode("S", "V")
    core("SaveAudio", ["audio", "filename_prefix"], "save", ["audio", "filename_prefix", "format"])().save("A", "p", "flac")
    assert calls == [("VAEEncode", {"vae": "V", "pixels": "P"}), ("VAEDecode", {"samples": "S", "vae": "V"}),
                     ("SaveAudio", {"audio": "A", "filename_prefix": "p"})]


def test_types_follow_into_the_modules_own_helpers():
    # PlusMinusTextClip: encode(self, clip, ...) -> self.encode_text(clip, text) -> clip.tokenize(text)
    new, _done, left = _rewrite('''
        def build(model, steps):
            return model.get_model_object("model_sampling")
        class PlusMinus:
            def encode(self, clip, positive, model):
                return self.encode_text(clip, positive), build(model, 4), self.other(clip), self.patch(model)
            def encode_text(self, clip, text):
                tokens = clip.tokenize(text)
                return clip.encode_from_tokens(tokens, return_pooled=True)
            def other(self, clip):
                clip = make_clip()
                return clip.tokenize("x")
            def patch(self, model):
                model = model.clone()
                return model
        ''', {"PlusMinus": ("encode", {"clip": "CLIP", "positive": "STRING", "model": "MODEL"})})
    assert "_v2e.sync(eng.clip.tokenize(clip, text))" in new
    assert "_v2e.sync(eng.clip.encode_from_tokens(clip, tokens, return_pooled=True))" in new
    assert 'return clip.tokenize("x")' in new  # reassigned in the helper: not followed
    assert "model = _v2e.sync(eng.model.clone(model))" in new  # a clone keeps the type
    assert "MODEL.get_model_object" in left


def test_nested_engine_calls_are_all_rewritten():
    # kaytool Strong_Prompt: clip.encode_from_tokens_scheduled(clip.tokenize(text)) kept the inner call
    new, _done, _left = _rewrite('''
        import comfy.model_management as mm
        class N:
            def run(self, clip, vae, text, image):
                cond = clip.encode_from_tokens_scheduled(clip.tokenize(text or ""))
                latent = vae.encode(image.to(mm.get_torch_device()))
                return cond, latent
        ''', {"N": ("run", {"clip": "CLIP", "vae": "VAE", "text": "STRING", "image": "IMAGE"})})
    assert '_v2e.sync(eng.clip.encode_tokens(clip, _v2e.sync(eng.clip.tokenize(clip, text or ""))))' in new
    assert "_v2e.sync(eng.vae.encode(vae, image.to(_v2e.get_torch_device())))" in new
    assert "clip.tokenize(text" not in new.replace("eng.clip.tokenize(clip, text", "")


def test_api_routes_called_through_fetch_api_are_moved(tmp_path):
    # LLM-Helper: routes.get("/api/llmhelper/models"); the frontend calls api.fetchApi("/llmhelper/models")
    pack = tmp_path / "pack"
    (pack / "web").mkdir(parents=True)
    (pack / "__init__.py").write_text(textwrap.dedent('''
        class N:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}
            RETURN_TYPES = ()
            FUNCTION = "run"
            def run(self):
                return ()
        NODE_CLASS_MAPPINGS = {"N": N}
        WEB_DIRECTORY = "./web"
        '''))
    (pack / "web" / "a.js").write_text('const r = await api.fetchApi("/llmhelper/models", {});\nfetch("/llmhelper/models");\n')
    describe = {"ok": True, "routes": [["GET", "/api/llmhelper/models"]],
                "nodes": {"N": {"class": "N", "kind": "v1", "function": "run", "info": {"input": {"required": {}}}}}}
    conversion = convert.generate(str(pack), describe, str(tmp_path / "out"))
    js = open(tmp_path / "out" / "pack" / "v2" / "web" / "a.js").read()
    namespace = conversion["namespace"]
    assert f'api.fetchApi("/ext/{namespace}/api/llmhelper/models"' in js
    assert 'fetch("/llmhelper/models")' in js  # without fetchApi there's no /api prefix: not this route
    assert conversion["routes"]["not_in_frontend"] == []


def test_async_node_code_awaits_engine_calls():
    new, _done, _left = _rewrite('''
        class Skim:
            @classmethod
            async def execute(cls, model, vae, pixels):
                m = model.clone()
                def later():
                    return vae.encode(pixels)
                return m, later
        ''', {"Skim": ("execute", {"model": "MODEL", "vae": "VAE", "pixels": "IMAGE"})})
    assert "m = (await eng.model.clone(model))" in new
    assert "return _v2e.sync(eng.vae.encode(vae, pixels))" in new  # a sync function inside: no await there


def test_v3_sync_execute_runs_with_engine_calls_routed():
    import asyncio
    import inspect
    source = convert.V3_ENTRY
    start = source.index("def _v2_engine_api(cls):")
    end = source.index("\n\n\ndef _v2_is_v3", start)
    ran = []

    class Shim:
        @staticmethod
        async def run_v3(function, cls, *args, **kwargs):
            ran.append(cls.__name__)
            return function(cls, *args, **kwargs)
    namespace = {"_v2_inspect": inspect, "_v2_shim": Shim}
    exec(source[start:end], namespace)  # noqa: S102

    class Sync:
        @classmethod
        def execute(cls, x):
            return x + 1

    class Async:
        @classmethod
        async def execute(cls, x):
            return x + 2
    wrapped_sync, wrapped_async = namespace["_v2_engine_api"](Sync), namespace["_v2_engine_api"](Async)
    assert wrapped_sync.ENGINE_API and inspect.iscoroutinefunction(wrapped_sync.execute)
    assert asyncio.run(wrapped_sync.execute(1)) == 2 and ran == ["Sync"]
    assert asyncio.run(wrapped_async.execute(1)) == 3 and ran == ["Sync"]  # async: as the pack wrote it


def test_core_node_stand_in_methods_have_the_methods_parameters():
    import inspect
    source = convert.ENGINE_SHIM
    start = source.index("class CoreNode:")
    end = source.index("\n\n\n", start)
    namespace = {"sync": lambda result: result,
                 "eng": type("E", (), {"node": type("N", (), {"run": staticmethod(lambda i, a: (i, a))})})}
    exec(source[start:end], namespace)  # noqa: S102
    encoder = namespace["CoreNode"]("VAEEncodeTiled", ["pixels", "vae", "tile_size", "overlap"], "encode",
                                    ["vae", "pixels", "tile_size", "overlap"])()
    assert "overlap" in inspect.signature(encoder.encode).parameters  # Impact's ReencodeLatent checks this
    assert encoder.encode("V", "P", 512, overlap=32) == ("VAEEncodeTiled", {"vae": "V", "pixels": "P",
                                                                            "tile_size": 512, "overlap": 32})


def test_v1_classes_listed_only_through_a_v3_entry_are_found(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "__init__.py").write_text("from . import tpg\nasync def comfy_entrypoint():\n    return None\n")
    (pack / "tpg.py").write_text("class TPG:\n    pass\n")
    nodes = {"TPG": {"kind": "v1", "module": "/pack.tpg", "qualname": "TPG"}}
    code = convert._v1_fallback(str(pack), nodes)
    assert "{'TPG': ('.tpg', 'TPG')}" in code
    (pack / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n")
    assert convert._v1_fallback(str(pack), nodes) == ""  # the table is there: nothing added, code as it was


def test_sampling_functions_reading_the_model_are_flagged():
    # sd-perturbed-attention PAG: post_cfg_function(args) reads args["model"], which pack code never gets
    _new, _done, left = _rewrite('''
        class PAG:
            def patch(self, model, scale):
                m = model.clone()
                def post_cfg_function(args):
                    inner = args["model"]
                    return args["denoised"] * scale + inner.x
                def pre(args):
                    return args.get("model")
                m.set_model_sampler_post_cfg_function(post_cfg_function)
                m.set_model_sampler_pre_cfg_function(pre)
                return (m,)
        ''', {"PAG": ("patch", {"model": "MODEL", "scale": "FLOAT"})})
    flagged = [x for x in left if x.startswith("sampling function")]
    assert set(flagged) == {"sampling function args['model'] (not passed to pack code)"}
    assert not any("denoised" in x for x in left)


def test_time_stamps_are_masked_but_other_text_is_not():
    import importlib
    verify = importlib.import_module("verify")
    a = ("Captured: 2026-11-07 07:35:38\nPython: 3.13", ["out_20261107_073538.png", 4])
    b = ("Captured: 2026-11-07 07:35:49\nPython: 3.13", ["out_20261107_073549.png", 4])
    assert verify._mask_times(a) == verify._mask_times(b)
    assert verify._mask_times("value 42") == "value 42"
    assert verify._mask_times(("Python: 3.13",)) != verify._mask_times(("Python: 3.12",))


def test_model_loading_by_path_goes_through_the_engine():
    # Image Saver / TinyTerra / Inspire: a checkpoint loaded by path in the pack's process came back as a PackValue
    new, _done, left = _rewrite('''
        import comfy.sd
        import folder_paths
        class Loader:
            def load(self, ckpt_name, unet_name):
                path = folder_paths.get_full_path("checkpoints", ckpt_name)
                out = comfy.sd.load_checkpoint_guess_config(path, output_vae=True, output_clip=True)
                unet = comfy.sd.load_diffusion_model(folder_paths.get_full_path("diffusion_models", unet_name))
                return (*out[:3], unet)
    ''')
    assert "_v2e.load_checkpoint_guess_config(path" in new and "_v2e.load_diffusion_model(" in new
    assert not left
    compile(convert.ENGINE_SHIM, "_v2_engine.py", "exec")  # the shim with the new loaders still compiles


def test_routes_nothing_in_the_pack_calls_are_told_apart(tmp_path):
    v2 = tmp_path / "v2"
    (v2 / "web").mkdir(parents=True)
    (v2 / "nodes.py").write_text('@routes.get("/mine/used")\nasync def a(r): pass\n@routes.get("/mine/outside")\n'
                                 'async def b(r): pass\n')
    (v2 / "web" / "app.js").write_text('const url = "/mine/" + "used";\nfetch(base + "/mine/used?x=" + id)\n')
    assert convert.route_called(str(v2), "/mine/used")       # built at run time in the frontend: still a manual item
    assert not convert.route_called(str(v2), "/mine/outside")  # only its own declaration: an outside caller, a note


def test_a_model_file_symlinked_from_elsewhere_is_in_its_folder(tmp_path):
    # Danny: checkpoints/x.safetensors -> /cache/<sha>; resolving the link first put it outside the folder
    import re as _re
    source = _re.search(r"\ndef _contained\(.*?\n    return None\n", convert.ENGINE_SHIM, _re.S).group(0)
    scope = {}
    exec(source, scope)  # noqa: S102
    contained = scope["_contained"]
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "abc123").write_bytes(b"w")
    folder = tmp_path / "models" / "checkpoints"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "x.safetensors").symlink_to(cache / "abc123")
    (folder / "plain.safetensors").write_bytes(b"w")
    assert contained(str(folder / "sub" / "x.safetensors"), [str(folder)]) == "sub/x.safetensors"
    assert contained(str(folder / "plain.safetensors"), [str(folder)]) == "plain.safetensors"
    assert contained(str(cache / "abc123"), [str(folder)]) is None            # outside every model folder
    assert contained(str(folder / ".." / ".." / "cache" / "abc123"), [str(folder)]) is None
    linked = tmp_path / "linked_models"
    linked.symlink_to(folder)                                               # a folder reached through a link
    assert contained(str(folder / "plain.safetensors"), [str(linked)]) == "plain.safetensors"


def test_attributes_of_every_engine_object_type_are_flagged():
    # WhiteRabbit UpscaleWithModelAdvanced: upscale_model.to(device) passed as perfect, then failed on a Ref (Danny)
    _new, _done, left = _rewrite('''
        class Up:
            def run(self, upscale_model, image, control_net):
                upscale_model.to("cuda")
                scale = upscale_model.scale
                return (control_net.copy(), image)
    ''', typed={"Up": ("run", {"upscale_model": "UPSCALE_MODEL", "image": "IMAGE", "control_net": "CONTROL_NET"})})
    assert "UPSCALE_MODEL.to" in left and "UPSCALE_MODEL.scale" in left and "CONTROL_NET.copy" in left
    assert not any(n.startswith("IMAGE.") for n in left)  # plain data types aren't engine objects


def test_a_callable_object_reading_the_model_is_flagged():
    # FDGuidance FDG_APG_Patcher: an object with __call__ on set_model_sampler_cfg_function reads args["model"]
    _new, _done, left = _rewrite('''
        class Calc:
            def __init__(self, p):
                self.p = p
            def __call__(self, args):
                return args["model"].model_sampling.timestep(args["sigma"])
        class Patch:
            def patch(self, model, scale):
                m = model.clone()
                calc = Calc(scale)
                model.set_model_sampler_cfg_function(calc)
                return (model,)
    ''', typed={"Patch": ("patch", {"model": "MODEL", "scale": "FLOAT"})})
    assert any("args['model']" in n for n in left)


def test_a_comfyui_import_in_a_class_body_keeps_working_as_an_attribute():
    # MakkiTools BatchLoraLoader (Danny r2): ``import folder_paths`` in the class body, used as self.folder_paths;
    # the converter dropped the import and the node failed
    new, _done, left = _rewrite('''
        class BatchLoraLoader:
            import folder_paths
            @classmethod
            def INPUT_TYPES(s):
                return {"required": {"lora": (s.folder_paths.get_filename_list("loras"),)}}
            def load(self, lora):
                return self.folder_paths.get_full_path_or_raise("loras", lora)
    ''')
    assert "from comfy_api.v4 import util as _v2_util; folder_paths = _v2_util.folders" in new
    assert "\n    import folder_paths" not in new
    assert not left
    compile(new, "nodes.py", "exec")


def test_more_clock_readings_and_the_pack_folder_are_masked():
    # the 24 "outputs differ" packs (2026-11-07): compact stamps, Unix time, time seeds, the pack's own path
    import importlib
    verify = importlib.import_module("verify")
    pairs = [("2026-11-07-223812", "2026-11-07-223819"), ("261107/x/261107223848_x", "261107/x/261107223855_x"),
             (1794091153, 1794091160), (20261108209700, 20261108209705),
             ("/tmp/out/1794091245_comic.html", "/tmp/out/1794091296_comic.html")]
    for a, b in pairs:
        assert verify._mask_times(a) == verify._mask_times(b), (a, b)
    roots = ("/pack", "/work/converted/ez/v2")
    assert verify._mask_times("/pack/data/x", 0, roots) == verify._mask_times("/work/converted/ez/v2/data/x", 0, roots)
    assert verify._mask_times(42) == 42 and verify._mask_times(True) is True
    assert verify._mask_times("value 1.5 and 12") == "value 1.5 and 12"


def test_the_shims_savers_pass_images_through_like_comfyuis():
    # SaveImageWithoutMetadata / promptloop / YCNodes subclass SaveImage or PreviewImage: ComfyUI's save_images
    # returns {"ui": ..., "result": (images,)}; the shim dropped "result" (verify: "output: 1 items vs 0")
    import re as _re
    source = _re.search(r"\ndef _plain_ui\(.*?\n    return out\n", convert.ENGINE_SHIM, _re.S).group(0)
    scope = {}
    exec(source, scope)  # noqa: S102
    images = object()
    assert scope["_plain_ui"]({"images": []}, images) == {"ui": {"images": []}, "result": (images,)}


def test_the_shims_savers_write_metadata_and_use_their_folder(monkeypatch):
    # Danny r7: PromptLoop's PNGs lost the prompt and workflow (cls=None); MaskPreview's preview went to output
    import enum
    import types
    start = convert.ENGINE_SHIM.index("def _plain_ui(")
    end = convert.ENGINE_SHIM.index("def _output(result):")
    calls = []
    class FolderType(str, enum.Enum):
        output = "output"
        temp = "temp"
    class ImageSaveHelper:
        @staticmethod
        def save_images(images, filename_prefix, folder_type, cls, compress_level=4):
            calls.append((filename_prefix, folder_type, cls.hidden.prompt, cls.hidden.extra_pnginfo, compress_level))
            return ["saved"]
    ui = types.SimpleNamespace(ImageSaveHelper=ImageSaveHelper, SavedImages=lambda saved: {"images": saved})
    index = types.ModuleType("comfy_api.v4")
    index.ui, index.io = ui, types.SimpleNamespace(FolderType=FolderType)
    monkeypatch.setitem(sys.modules, "comfy_api.v4", index)
    scope = {}
    exec(convert.ENGINE_SHIM[start:end], scope)  # noqa: S102
    images = object()
    out = scope["SaveImage"]().save_images(images, "loop", {"1": {}}, {"workflow": {"nodes": []}})
    assert out == {"ui": {"images": ["saved"]}, "result": (images,)}
    assert calls[-1] == ("loop", FolderType.output, {"1": {}}, {"workflow": {"nodes": []}}, 4)

    class MaskPreview(scope["PreviewImage"]):
        pass
    MaskPreview().save_images(images, "mask", [{"1": {}}], [{"workflow": {}}])  # INPUT_IS_LIST hidden lists
    prefix, folder, prompt, info, level = calls[-1]
    assert prefix.startswith("mask_temp_") and folder == FolderType.temp and level == 1
    assert prompt == {"1": {}} and info == {"workflow": {}}

    class NoInit(scope["PreviewImage"]):
        def __init__(self):
            pass  # a subclass that doesn't call super().__init__: the class defaults still apply
    NoInit().save_images(images)
    assert calls[-1][1] == FolderType.temp


def test_hasattr_on_an_engine_object_is_flagged():
    # AK-XZ-Axis Batch Prompts (Danny r3, agent edit): hasattr(clip, "tokenize") is always False on a V2.1 handle
    _new, _done, left = _rewrite('''
        def _encode(clip, text):
            if hasattr(clip, "tokenize") and hasattr(clip, "encode_from_tokens"):
                return 1
            return getattr(clip, "cond_stage_model", None)
        def _other(model, x):  # a param named like an engine object, but not passed one: not flagged
            return hasattr(model, "generate")
        class Node:
            def run(self, model, clip, text, width):
                if hasattr(model, "model_sampling") and hasattr(text, "upper"):
                    return (model, _encode(clip, text))
    ''', typed={"Node": ("run", {"model": "MODEL", "clip": "CLIP", "text": "STRING", "width": "INT"})})
    assert "CLIP.tokenize (hasattr on a V2.1 handle)" in left
    assert "CLIP.cond_stage_model (getattr on a V2.1 handle)" in left
    assert "MODEL.model_sampling (hasattr on a V2.1 handle)" in left
    assert not any(n.startswith("STRING.") for n in left)
    assert "MODEL.generate (hasattr on a V2.1 handle)" not in left


def test_hasattr_on_a_type_the_pack_itself_outputs_isnt_flagged():
    # TTS/VLM packs: their loader node outputs their own model as "MODEL"; it stays in the pack's process
    _new, _done, left = _rewrite('''
        class Speak:
            def run(self, model, text):
                return (hasattr(model, "device"),)
    ''', typed={"Speak": ("run", {"model": "MODEL", "text": "STRING"}), convert.OWNED_TYPES_KEY: (None, frozenset({"MODEL"}))})
    assert not left


def test_a_sampling_hook_returned_by_a_factory_is_checked():
    # DiversityBoost (Danny r3): diversity_hook reads args["model"]; it's returned and registered elsewhere
    _new, _done, left = _rewrite('''
        def make_hook(strength):
            def diversity_hook(args):
                denoised = args["denoised"]
                model = args["model"]
                return denoised * strength
            return diversity_hook
        def unrelated(config):
            return config["model"]  # a config dict, not sampling args: not flagged
    ''')
    assert "sampling function args['model'] (not passed to pack code)" in left
    assert len([n for n in left if "args['model']" in n]) == 1


def test_the_shims_lora_loader_takes_comfyuis_lora_metadata():
    # LoRABlockFilter (Danny r4): load_lora_for_models(..., lora_metadata=meta) raised TypeError on the shim
    import inspect
    import re as _re
    source = _re.search(r"\ndef load_lora_for_models\(.*?\n\n\n", convert.ENGINE_SHIM, _re.S).group(0)
    scope = {}
    exec(source, scope)  # noqa: S102
    assert "lora_metadata" in inspect.signature(scope["load_lora_for_models"]).parameters


def test_comfyui_modules_imported_at_run_time_and_attention_patch_model_reads_are_flagged():
    # SodaLoadStoryboardLoRA: importlib.import_module("comfy.sd"); DepthDiff: extra_options["model"] (Danny r4)
    _new, _done, left = _rewrite('''
        import importlib
        def load(path):
            sd = importlib.import_module("comfy.sd")
            return sd.load_lora_for_models
        def attn2(q, k, v, extra_options):
            shape = extra_options["original_shape"]
            return extra_options["model"]
        def plain():
            return importlib.import_module("json")
    ''')
    assert "comfy.sd (imported at run time)" in left
    assert "sampling function args['model'] (not passed to pack code)" in left
    assert not any("json" in n for n in left)


def test_only_types_a_pack_creates_count_as_its_own():
    # SeamlessTile (Danny r5) takes ComfyUI's MODEL and returns it patched: getattr(model, "clone") must be flagged;
    # a TTS loader that outputs its own model as "MODEL" with no MODEL input owns that type
    patch = {"SeamlessTile": {"info": {"input": {"required": {"model": ["MODEL"], "tiling": [["a", "b"]]}}, "output": ["MODEL"]}}}
    loader = {"TTSLoader": {"info": {"input": {"required": {"name": [["x"]]}}, "output": ["MODEL"]}}}
    assert convert.owned_types(patch) == frozenset()
    assert convert.owned_types(loader) == frozenset({"MODEL"})


def test_a_pack_changing_pytorchs_process_wide_settings_is_flagged():
    # CachingCLIPTextEncode (Danny r5): sets TF32 at import; in ComfyUI that changed every node's math
    _new, _done, left = _rewrite('''
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        class Encode:
            def run(self, clip, text):
                return (clip.encode_from_tokens(clip.tokenize(text)),)
    ''', typed={"Encode": ("run", {"clip": "CLIP", "text": "STRING"})})
    assert any("torch.backends.cuda.matmul.allow_tf32" in n for n in left)
    assert any("torch.set_float32_matmul_precision" in n for n in left)
    # a pack whose own model runs in its own process (SAM2, BiRefNet): its setting still applies there; not flagged
    _new, _done, left = _rewrite('''
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        def run_own_model(net, x):
            return net(x)
    ''')
    assert not left


def test_an_input_is_list_v1_node_gets_its_hidden_inputs_as_lists():
    # ShowText (Danny r6): INPUT_IS_LIST = True reads extra_pnginfo[0] / unique_id[0], as ComfyUI passes them
    assert "inputs[arg] = [value] if as_list else value" in convert.ENGINE_SHIM
    assert 'as_list = bool(getattr(v1_class, "INPUT_IS_LIST", False))' in convert.ENGINE_SHIM


def test_calling_an_engine_handle_directly_is_flagged():
    # MambaIR (Danny r4): tiled upscaling calls the UPSCALE_MODEL itself inside a lambda
    _new, _done, left = _rewrite('''
        class Up:
            def run(self, upscale_model, image):
                f = lambda a: upscale_model(a)
                return (f(image),)
    ''', typed={"Up": ("run", {"upscale_model": "UPSCALE_MODEL", "image": "IMAGE"})})
    assert "UPSCALE_MODEL called directly (a V2.1 handle isn't callable)" in left


def test_changing_the_workflow_metadata_in_place_is_flagged():
    # ShowText (Danny r7): a node's widgets_values in extra_pnginfo's workflow changed; a pack has a copy
    msg = "a workflow node changed in place via extra_pnginfo (the saved workflow won't see it)"
    for code in ('''
        def notify(text, unique_id=None, extra_pnginfo=None):
            extra_pnginfo[0]["workflow"]["nodes"][0]["widgets_values"] = [text]
            return text
    ''', '''
        def through_aliases(text, extra_pnginfo=None):
            workflow = extra_pnginfo[0]["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == "1"), None)
            node["widgets_values"] = [text]
    ''', '''
        def loop_alias(seed, extra_pnginfo=None):
            for i, n in enumerate(extra_pnginfo.get("workflow", {}).get("nodes", [])):
                n["widgets_values"][0] = seed
    ''', '''
        def append(text, extra_pnginfo=None):
            for n in extra_pnginfo["workflow"]["nodes"]:
                n["widgets_values"].append(text)
    '''):
        _new, _done, left = _rewrite(code)
        assert msg in left, code


def test_a_saver_building_its_own_metadata_is_not_flagged():
    _new, _done, left = _rewrite('''
        import json
        def read_only(extra_pnginfo=None):
            workflow = extra_pnginfo["workflow"]
            return len(workflow["nodes"])
        def save(images, prompt=None, extra_pnginfo=None):
            metadata = {}
            exif = {}
            for key, value in extra_pnginfo.items():
                exif[key] = json.dumps(value)
            info = dict(extra_pnginfo)
            info["prompt"] = prompt
            save_kwargs = {"pnginfo": extra_pnginfo}
            save_kwargs["exif"] = exif
            copy = json.loads(json.dumps(extra_pnginfo))
            copy["workflow"]["nodes"][0]["widgets_values"] = []
            return metadata
    ''')
    assert not left


def test_the_entry_leaves_comfyuis_loader_to_call_comfy_entrypoint(tmp_path):
    # ComfyUI's load_custom_node reads a V1 table that isn't None and returns, never calling comfy_entrypoint: the
    # entry's old ``NODE_CLASS_MAPPINGS = {}`` registered no node when ComfyUI loaded a converted pack in process
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "__init__.py").write_text(textwrap.dedent('''
        class Echo:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"text": ("STRING", {})}}
            RETURN_TYPES = ("STRING",)
            FUNCTION = "run"
            def run(self, text):
                return (text,)
        NODE_CLASS_MAPPINGS = {"Echo": Echo}
        NODE_DISPLAY_NAME_MAPPINGS = {"Echo": "Echo"}
        '''))
    describe = {"ok": True, "nodes": {"Echo": {"class": "Echo", "kind": "v1", "module": str(pack),
                                               "qualname": "Echo", "info": {"input": {"required": {"text": ["STRING", {}]}}}}}}
    convert.generate(str(pack), describe, str(tmp_path / "out"))
    tree = ast.parse(open(tmp_path / "out" / "pack" / "v2" / "__init__.py").read())
    final = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            final[node.targets[0].id] = node.value
    assert isinstance(final["NODE_CLASS_MAPPINGS"], ast.Constant) and final["NODE_CLASS_MAPPINGS"].value is None
    assert isinstance(final["NODE_DISPLAY_NAME_MAPPINGS"], ast.Constant) and final["NODE_DISPLAY_NAME_MAPPINGS"].value is None
    assert any(isinstance(n, ast.AsyncFunctionDef) and n.name == "comfy_entrypoint" for n in tree.body)


def test_a_modules_own_util_does_not_hide_v2s():
    # D2 nodes: ``from .modules import util`` hid V2's util, so util.SAMPLER_NAMES failed and the whole pack didn't load
    new, _done, _left = convert.rewrite(textwrap.dedent("""
        import comfy.samplers
        from .modules import util
        def options():
            return comfy.samplers.KSampler.SAMPLERS, util.mine()
    """), "nodes.py", {})
    assert "_v2util.SAMPLER_NAMES" in new and "util.mine()" in new
    assert "from comfy_api.v4 import util as _v2util" in new
    ast.parse(new)


def test_a_module_without_its_own_util_is_unchanged():
    new, _done, _left = convert.rewrite(textwrap.dedent("""
        import comfy.samplers
        def options():
            return comfy.samplers.KSampler.SAMPLERS
    """), "nodes.py", {})
    assert "_v2util" not in new and "return util.SAMPLER_NAMES" in new


def test_comfyuis_vae_list_needs_no_engine_call():
    # D2 nodes: define_schema called VAELoader.vae_list, which became an engine call outside a node: the pack didn't load
    new, _done, left = _rewrite("""
        import nodes
        from nodes import VAELoader
        def a():
            return ["none"] + VAELoader.vae_list(VAELoader)
        def b():
            return nodes.VAELoader.vae_list(nodes.VAELoader)
    """)
    assert new.count("_v2e.vae_list()") == 2 and "CoreNode" not in new
    assert "nodes.VAELoader.vae_list" not in " ".join(left)


def test_the_shims_vae_list_matches_comfyuis_rules(monkeypatch):
    import types
    start = convert.ENGINE_SHIM.index("def vae_list():")
    end = convert.ENGINE_SHIM.index("def _hidden_as_cls(")
    files = {"vae": ["a.safetensors"], "vae_approx": ["taesd_encoder.pth", "taesd_decoder.pth", "taesdxl_decoder.pth",
                                                      "taehv.pth", "other.pth"]}
    index = types.ModuleType("comfy_api.v4")
    index.util = types.SimpleNamespace(folders=types.SimpleNamespace(get_filename_list=lambda f: files[f]))
    monkeypatch.setitem(sys.modules, "comfy_api.v4", index)
    scope = {}
    exec(convert.ENGINE_SHIM[start:end], scope)  # noqa: S102
    assert scope["vae_list"]() == ["a.safetensors", "taehv.pth", "taesd", "pixel_space"]


def test_the_verifier_runs_a_v3_nodes_plain_execute():
    # 2,122 V3 nodes were never compared: the verifier awaited a plain execute's NodeOutput
    import asyncio
    import inspect
    source = open(os.path.join(os.path.dirname(convert.__file__), "verify.py")).read()
    assert "inspect.isawaitable(result)" in source and "return await clone.execute(**inputs)" not in source
    async def call(fn):
        result = fn()
        return (await result) if inspect.isawaitable(result) else result
    async def later():
        return 2
    assert asyncio.run(call(lambda: 1)) == 1 and asyncio.run(call(later)) == 2


def test_v3_source_location_and_execute_parameters_are_recorded():
    import describe

    class Example:
        FUNCTION = "EXECUTE_NORMALIZED"

        @classmethod
        def execute(cls, width, height=512):
            return width, height

    info = describe.method_info(Example, "execute")
    assert info["function"] == "execute"
    assert info["params"] == ["width", "height"]
    assert info["file"] == __file__
    assert info["qualname"].endswith("Example")


@pytest.mark.parametrize("async_function", [False, True])
def test_named_sampler_and_model_sigmas_use_existing_engine_calls(async_function):
    import asyncio
    import types

    prefix = "async " if async_function else ""
    new, done, left = _rewrite(f'''
        import comfy.samplers
        class Node:
            {prefix}def execute(self, model, scheduler, steps):
                sampler = comfy.samplers.sampler_object("euler")
                sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)
                return sampler, sigmas
    ''', {"Node": ("execute", {"model": "MODEL"})})
    assert not left
    assert "comfy.samplers.sampler_object" in done
    assert "comfy.samplers.calculate_sigmas" in done
    calls = []

    async def run(name, inputs):
        calls.append((name, inputs))
        return ["SAMPLER" if name == "KSamplerSelect" else "SIGMAS"]

    scope = {"eng": types.SimpleNamespace(node=types.SimpleNamespace(run=run)),
             "_v2e": types.SimpleNamespace(sync=asyncio.run)}
    # Run the generated algorithm without importing a pack or a real engine.
    tree = ast.parse(new)
    classes = ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[])
    exec(compile(classes, "converted", "exec"), scope)  # noqa: S102
    model = object()
    result = scope["Node"]().execute(model, "normal", 20)
    if async_function:
        result = asyncio.run(result)
    assert result == ("SAMPLER", "SIGMAS")
    assert calls == [("KSamplerSelect", {"sampler_name": "euler"}), ("BasicScheduler", {
        "model": model, "scheduler": "normal", "steps": 20, "denoise": 1.0})]


def test_other_model_objects_are_not_mistaken_for_the_sigma_schedule():
    new, _done, left = _rewrite('''
        import comfy.samplers
        class Node:
            def execute(self, model):
                return comfy.samplers.calculate_sigmas(model.get_model_object("diffusion_model"), "normal", 20)
    ''', {"Node": ("execute", {"model": "MODEL"})})
    assert "BasicScheduler" not in new
    assert "MODEL.get_model_object" in left


def test_named_sampler_alias_and_keyword_keep_special_sampler_names():
    new, _done, left = _rewrite('''
        from comfy.samplers import sampler_object as select
        def run():
            return select(name="uni_pc"), select("ddim")
    ''')
    assert not left
    assert "{'sampler_name': \"uni_pc\"}" in new
    assert "{'sampler_name': \"ddim\"}" in new
    assert "eng.sampler.named" not in new

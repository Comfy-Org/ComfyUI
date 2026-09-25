"""Unit tests for the wrapper API workflow builder, model requirements and
OpenAPI spec. The graph validation test at the end only runs when torch is
available (it loads the real node registry)."""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api_wrapper import openapi as wrapper_openapi
from api_wrapper import workflows as wrapper_workflows
from api_wrapper.workflows import (
    FLUX2_KLEIN_9B_CLIP,
    FLUX2_KLEIN_9B_MODELS,
    FLUX2_KLEIN_9B_UNET,
    FLUX2_KLEIN_9B_VAE,
    build_flux2_klein_9b_img2img,
)

EXPECTED_CLASS_TYPES = {
    "UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "ImageScaleToTotalPixels",
    "GetImageSize", "CLIPTextEncode", "VAEEncode", "ReferenceLatent", "CFGGuider",
    "KSamplerSelect", "Flux2Scheduler", "RandomNoise", "EmptyFlux2LatentImage",
    "SamplerCustomAdvanced", "VAEDecode", "SaveImage",
}


def _sample_graph():
    return build_flux2_klein_9b_img2img(
        prompt="make it snow",
        image="wrapper/abc123.png",
        negative_prompt="blurry",
        seed=42,
        steps=25,
        cfg=4.5,
        megapixels=1.5,
    )


class TestWorkflowGraph(unittest.TestCase):
    def test_node_count_and_class_types(self):
        graph = _sample_graph()
        self.assertEqual(len(graph), 19)
        self.assertEqual({node["class_type"] for node in graph.values()}, EXPECTED_CLASS_TYPES)

    def test_links_reference_existing_nodes(self):
        graph = _sample_graph()
        for node_id, node in graph.items():
            for input_name, value in node["inputs"].items():
                if isinstance(value, list):
                    src_id, src_index = value
                    self.assertIn(src_id, graph, f"{node_id}.{input_name} links to unknown node {src_id}")
                    self.assertIsInstance(src_index, int)
                    self.assertGreaterEqual(src_index, 0)

    def test_output_node_present(self):
        graph = _sample_graph()
        output_nodes = [n for n in graph.values() if n["class_type"] == "SaveImage"]
        self.assertEqual(len(output_nodes), 1)
        self.assertIsInstance(output_nodes[0]["inputs"]["images"], list)

    def test_loader_filenames_match_model_requirements(self):
        graph = _sample_graph()
        unet = next(n for n in graph.values() if n["class_type"] == "UNETLoader")
        clip = next(n for n in graph.values() if n["class_type"] == "CLIPLoader")
        vae = next(n for n in graph.values() if n["class_type"] == "VAELoader")
        self.assertEqual(unet["inputs"]["unet_name"], FLUX2_KLEIN_9B_UNET)
        self.assertEqual(clip["inputs"]["clip_name"], FLUX2_KLEIN_9B_CLIP)
        self.assertEqual(clip["inputs"]["type"], "flux2")
        self.assertEqual(vae["inputs"]["vae_name"], FLUX2_KLEIN_9B_VAE)
        filenames = {m["filename"] for m in FLUX2_KLEIN_9B_MODELS}
        self.assertEqual(filenames, {FLUX2_KLEIN_9B_UNET, FLUX2_KLEIN_9B_CLIP, FLUX2_KLEIN_9B_VAE})

    def test_sampler_settings(self):
        graph = _sample_graph()
        guider = next(n for n in graph.values() if n["class_type"] == "CFGGuider")
        scheduler = next(n for n in graph.values() if n["class_type"] == "Flux2Scheduler")
        noise = next(n for n in graph.values() if n["class_type"] == "RandomNoise")
        self.assertEqual(guider["inputs"]["cfg"], 4.5)
        self.assertEqual(scheduler["inputs"]["steps"], 25)
        self.assertEqual(noise["inputs"]["noise_seed"], 42)


class TestKleinEditReferences(unittest.TestCase):
    def test_several_references_chain_one_reference_latent_each(self):
        graph = build_flux2_klein_9b_img2img(
            prompt="the same man, at the bus stop", ref_images=["wrapper/face.png", "wrapper/stop.png"],
            width=1024, height=576)
        self.assertEqual(graph["4"]["inputs"]["image"], "wrapper/face.png")
        self.assertEqual(graph["20"]["inputs"]["image"], "wrapper/stop.png")
        # face -> stop, on both sides of the guidance
        self.assertEqual(graph["23"]["inputs"], {"conditioning": ["10", 0], "latent": ["22", 0]})
        self.assertEqual(graph["24"]["inputs"], {"conditioning": ["11", 0], "latent": ["22", 0]})
        self.assertEqual(graph["12"]["inputs"]["positive"], ["23", 0])
        self.assertEqual(graph["12"]["inputs"]["negative"], ["24", 0])
        # the canvas is the one asked for, not the first reference's
        self.assertEqual(graph["16"]["inputs"]["width"], 1024)
        self.assertEqual(graph["14"]["inputs"]["height"], 576)
        self.assertEqual(graph["5"]["inputs"]["upscale_method"], "lanczos")

    def test_image_and_ref_images_combine_image_first(self):
        graph = build_flux2_klein_9b_img2img(prompt="x", image="wrapper/a.png", ref_images="wrapper/b.png")
        self.assertEqual(graph["4"]["inputs"]["image"], "wrapper/a.png")
        self.assertEqual(graph["20"]["inputs"]["image"], "wrapper/b.png")
        self.assertEqual(graph["16"]["inputs"]["width"], ["6", 0])  # no canvas: first image's size

    def test_no_image_at_all_is_refused(self):
        with self.assertRaises(ValueError):
            build_flux2_klein_9b_img2img(prompt="x")

    def test_either_upload_field_satisfies_the_edit(self):
        entry = wrapper_workflows.WORKFLOWS["flux2klein9b"]
        self.assertEqual(entry["image_fields"], ["image", "ref_images"])
        self.assertEqual(entry["uploads"]["ref_images"]["max"], 3)


class TestModelRequirements(unittest.TestCase):
    def test_three_models_with_folders_and_urls(self):
        self.assertEqual(len(FLUX2_KLEIN_9B_MODELS), 3)
        for model in FLUX2_KLEIN_9B_MODELS:
            self.assertIn(model["folder"], {"diffusion_models", "text_encoders", "vae"})
            self.assertTrue(model["filename"].endswith(".safetensors"))
            self.assertTrue(model["url"].startswith("https://huggingface.co/"))


class TestWorkflowRegistry(unittest.TestCase):
    def test_flux2klein9b_registered(self):
        from api_wrapper import workflows as workflows_module
        self.assertIn("flux2klein9b", workflows_module.WORKFLOWS)
        entry = workflows_module.WORKFLOWS["flux2klein9b"]
        self.assertTrue(entry["requires_image"])
        self.assertIs(entry["build"], workflows_module.build_flux2_klein_9b_img2img)

    def test_ideogram4_registered(self):
        from api_wrapper import workflows as workflows_module
        self.assertIn("ideogram4", workflows_module.WORKFLOWS)
        entry = workflows_module.WORKFLOWS["ideogram4"]
        self.assertFalse(entry["requires_image"])
        self.assertIs(entry["build"], workflows_module.build_ideogram4_text2img)
        self.assertIn("high_level_description", entry["example_prompt"])
        self.assertIn("mode", entry["extra_form_properties"])

    def test_minimaxh3_registered(self):
        from api_wrapper import workflows as workflows_module
        self.assertIn("minimaxh3", workflows_module.WORKFLOWS)
        entry = workflows_module.WORKFLOWS["minimaxh3"]
        self.assertEqual(sorted(entry["tasks"]), ["image", "reference", "text"])
        self.assertEqual(entry["output_type"], "video")
        self.assertIn("uploads", entry["tasks"]["reference"])
        self.assertEqual(entry["tasks"]["reference"]["uploads"]["ref_images"]["max"], 3)
        self.assertEqual(entry["tasks"]["reference"]["uploads"]["ref_videos"]["max"], 1)
        self.assertEqual(entry["tasks"]["reference"]["uploads"]["ref_video_audios"]["max"], 1)
        self.assertEqual(entry["tasks"]["reference"]["uploads"]["ref_audios"]["max"], 2)

    def test_klein_txt2img_registered(self):
        from api_wrapper import workflows as workflows_module
        self.assertIn("flux2klein9b-txt2img", workflows_module.WORKFLOWS)
        entry = workflows_module.WORKFLOWS["flux2klein9b-txt2img"]
        self.assertFalse(entry["requires_image"])
        self.assertIs(entry["build"], workflows_module.build_flux2_klein_9b_text2img)
        self.assertIn("width", entry["extra_form_properties"])


class TestOpenAPISpec(unittest.TestCase):
    def test_spec_has_wrapper_paths(self):
        spec = wrapper_openapi.WRAPPER_OPENAPI_SPEC
        self.assertEqual(spec["openapi"], "3.0.3")
        paths = spec["paths"]
        self.assertIn("/api/wrapper/{workflow}/generate", paths)
        self.assertIn("/api/wrapper/workflows", paths)
        self.assertIn("/api/wrapper/jobs/{job_id}", paths)
        self.assertIn("/api/wrapper/jobs/{job_id}/image", paths)
        generate = paths["/api/wrapper/{workflow}/generate"]["post"]
        self.assertEqual(generate["requestBody"]["content"]["multipart/form-data"]["schema"]["required"],
                         ["prompt", "image"])
        form = generate["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"]
        self.assertIn("vram", form)
        self.assertEqual(form["vram"]["enum"], ["auto", "low", "normal", "high"])

    def test_minimax_nvfp4_in_expanded_spec(self):
        from api_wrapper import workflows as workflows_module
        spec = wrapper_openapi.spec_with_workflows(workflows_module.WORKFLOWS)
        minimax = spec["paths"]["/api/wrapper/minimaxh3/text/generate"]["post"]
        quantization = minimax["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"]["quantization"]
        self.assertEqual(quantization["enum"], ["fp8", "int8", "bf16", "nvfp4"])
        # the vram field survives per-workflow filtering for every workflow
        for path in ("/api/wrapper/flux2klein9b/generate", "/api/wrapper/ideogram4/generate",
                     "/api/wrapper/minimaxh3/image/generate"):
            props = spec["paths"][path]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"]
            self.assertIn("vram", props)

    def test_spec_expands_to_concrete_workflow_paths(self):
        from api_wrapper import workflows as workflows_module
        spec = wrapper_openapi.spec_with_workflows({"flux2klein9b": workflows_module.WORKFLOWS["flux2klein9b"],
                                                    "video9b": {"build": None, "title": "x"}})
        paths = spec["paths"]
        self.assertNotIn("/api/wrapper/{workflow}/generate", paths)
        self.assertIn("/api/wrapper/flux2klein9b/generate", paths)
        self.assertIn("/api/wrapper/video9b/generate", paths)
        flux2 = paths["/api/wrapper/flux2klein9b/generate"]["post"]
        self.assertEqual([p.get("name") for p in flux2.get("parameters", [])], [])  # no leftover {workflow} param

    def test_spec_folds_workflow_example_and_extra_fields(self):
        from api_wrapper import workflows as workflows_module
        spec = wrapper_openapi.spec_with_workflows({"ideogram4": workflows_module.WORKFLOWS["ideogram4"]})
        operation = spec["paths"]["/api/wrapper/ideogram4/generate"]["post"]
        schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertIn("high_level_description", schema["properties"]["prompt"]["description"])
        self.assertIn("mode", schema["properties"])
        self.assertEqual(schema["properties"]["mode"]["enum"], ["default", "quality", "turbo"])
        # ideogram4 is text-to-image: no image field, image not required
        self.assertNotIn("image", schema["properties"])
        self.assertEqual(schema["required"], ["prompt"])

    def test_spec_keeps_image_required_only_for_image_workflows(self):
        from api_wrapper import workflows as workflows_module
        spec = wrapper_openapi.spec_with_workflows(workflows_module.WORKFLOWS)
        flux2 = spec["paths"]["/api/wrapper/flux2klein9b/generate"]["post"]
        flux2_schema = flux2["requestBody"]["content"]["multipart/form-data"]["schema"]
        # image OR ref_images: one-of, so neither is required on its own
        self.assertEqual(flux2_schema["required"], ["prompt"])
        self.assertIn("megapixels", flux2_schema["properties"])
        self.assertIn("ref_images", flux2_schema["properties"])
        i2v = spec["paths"]["/api/wrapper/minimaxh3/image/generate"]["post"]
        i2v_schema = i2v["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertIn("image", i2v_schema["required"])  # a single image field stays required

    def test_spec_emits_task_paths_and_video_response(self):
        from api_wrapper import workflows as workflows_module
        spec = wrapper_openapi.spec_with_workflows(workflows_module.WORKFLOWS)
        for path in ("/api/wrapper/minimaxh3/text/generate",
                     "/api/wrapper/minimaxh3/image/generate",
                     "/api/wrapper/minimaxh3/reference/generate"):
            self.assertIn(path, spec["paths"], f"missing {path}")
        # the bare /{name}/{task} alias was removed as redundant
        self.assertNotIn("/api/wrapper/minimaxh3/reference", spec["paths"])
        op = spec["paths"]["/api/wrapper/minimaxh3/reference/generate"]["post"]
        content = op["responses"]["200"]["content"]
        self.assertEqual(list(content)[0], "video/mp4")
        schema = op["requestBody"]["content"]["multipart/form-data"]["schema"]
        # prompt/raw_prompt are one-of on a rewriting task, so neither is required
        self.assertEqual(schema["required"], [])
        for prop in ("prompt", "raw_prompt", "llm_image", "llm_model",
                     "ref_images", "ref_videos", "ref_video_audios", "ref_audios",
                     "ref_image_size", "duration"):
            self.assertIn(prop, schema["properties"])
        self.assertNotIn("image", schema["properties"])
        self.assertEqual(schema["properties"]["quantization"]["enum"], ["fp8", "int8", "bf16", "nvfp4"])
        # the image task requires an image upload and returns video
        op = spec["paths"]["/api/wrapper/minimaxh3/image/generate"]["post"]
        schema = op["requestBody"]["content"]["multipart/form-data"]["schema"]
        self.assertEqual(schema["required"], ["image"])
        self.assertEqual(list(op["responses"]["200"]["content"])[0], "video/mp4")

    def test_generate_response_is_an_image(self):
        """The 200 response must list the workflow's media type first so
        Swagger UI shows the artifact inline instead of a generic download."""
        from api_wrapper import workflows as workflows_module
        from api_wrapper.openapi import OUTPUT_MEDIA_TYPES
        spec = wrapper_openapi.spec_with_workflows(workflows_module.WORKFLOWS)
        for path, path_item in spec["paths"].items():
            if not path.endswith("/generate") or "post" not in path_item:
                continue
            workflow_name = path.split("/")[3]
            entry = workflows_module.WORKFLOWS[workflow_name]
            expected = OUTPUT_MEDIA_TYPES.get(entry.get("output_type"), "image/png")
            content = path_item["post"]["responses"]["200"]["content"]
            self.assertEqual(list(content)[0], expected,
                             f"{path}: expected {expected} first, got {list(content)}")

    def test_spec_paths_are_valid_openapi_shape(self):
        """Paths must map to HTTP-method-keyed path items (Swagger UI silently
        drops operations that are not wrapped in a method key)."""
        from api_wrapper import workflows as workflows_module
        spec = wrapper_openapi.spec_with_workflows(workflows_module.WORKFLOWS)
        operation_ids = []
        for path, path_item in spec["paths"].items():
            for method in ("get", "post"):
                if method in path_item:
                    operation = path_item[method]
                    operation_ids.append(operation["operationId"])
                    # concrete paths must not declare the removed {workflow} param
                    self.assertNotIn("workflow", [p.get("name") for p in operation.get("parameters", [])])
        self.assertEqual(len(operation_ids), len(set(operation_ids)),
                         f"duplicate operationIds: {operation_ids}")
        self.assertIn("generateFlux2klein9b", operation_ids)
        self.assertIn("generateIdeogram4", operation_ids)

    def test_swagger_html_points_at_wrapper_spec(self):
        self.assertIn("/api/wrapper/openapi.json", wrapper_openapi.WRAPPER_SWAGGER_HTML)
        self.assertIn("swagger-ui", wrapper_openapi.WRAPPER_SWAGGER_HTML)


class TestKleinTxt2imgGraph(unittest.TestCase):
    def test_structure_and_sampler(self):
        graph = wrapper_workflows.build_flux2_klein_9b_text2img(
            prompt="a cat", negative_prompt="blurry", seed=7, steps=25, cfg=4.5,
            width=512, height=384)
        self.assertEqual(len(graph), 13)
        ids = set(graph)
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], ids)
        self.assertNotIn("LoadImage", [n["class_type"] for n in graph.values()])
        self.assertEqual(graph["4"]["inputs"]["text"], "a cat")
        self.assertEqual(graph["5"]["inputs"]["text"], "blurry")
        self.assertEqual(graph["6"]["inputs"]["cfg"], 4.5)
        self.assertEqual(graph["8"]["inputs"]["steps"], 25)
        self.assertEqual(graph["8"]["inputs"]["width"], 512)
        self.assertEqual(graph["10"]["inputs"]["height"], 384)
        self.assertEqual(graph["9"]["inputs"]["noise_seed"], 7)

    def test_defaults(self):
        graph = wrapper_workflows.build_flux2_klein_9b_text2img(prompt="x")
        self.assertEqual(graph["8"]["inputs"]["width"], 1024)
        self.assertEqual(graph["10"]["inputs"]["height"], 1024)
        self.assertEqual(graph["13"]["inputs"]["filename_prefix"], "wrapper/flux2klein9b_txt2img")


class TestMiniMaxH3Graph(unittest.TestCase):
    def test_text_to_video_structure(self):
        graph = wrapper_workflows.build_minimax_h3_text_to_video(
            prompt="a cat walks", seed=3, steps=40, width=1344, height=768,
            duration=5.0, scheduler="beta")
        ids = set(graph)
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], ids)
        self.assertEqual(graph["1"]["class_type"], "UNETLoader")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], wrapper_workflows.MINIMAX_H3_UNET_INT8)
        self.assertEqual(graph["3"]["inputs"]["type"], "minimax")
        self.assertEqual(graph["6"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertEqual(graph["6"]["inputs"]["length"], 120)  # 5s * 24fps
        self.assertEqual(graph["7"]["class_type"], "BasicGuider")
        self.assertEqual(graph["9"]["class_type"], "KSamplerSelect")
        self.assertEqual(graph["9"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(graph["10"]["inputs"]["steps"], 40)
        self.assertEqual(graph["13"]["class_type"], "VAEDecodeAudio")
        self.assertEqual(graph["14"]["class_type"], "CreateVideo")
        self.assertEqual(graph["14"]["inputs"]["fps"], 24.0)
        self.assertEqual(graph["14"]["inputs"]["audio"], ["13", 0])
        self.assertEqual(graph["15"]["class_type"], "SaveVideo")
        self.assertEqual(graph["15"]["inputs"]["video"], ["14", 0])
        self.assertEqual(graph["15"]["inputs"]["filename_prefix"], "wrapper/minimaxh3_t2v")

    def test_image_to_video_wires_frames(self):
        graph = wrapper_workflows.build_minimax_h3_image_to_video(
            prompt="x", first_frame="wrapper/a.png", last_frame="wrapper/b.png")
        self.assertEqual(graph["7"]["class_type"], "LoadImage")
        self.assertEqual(graph["8"]["class_type"], "LoadImage")
        # each frame is cover-cropped to the canvas before the node sees it
        self.assertEqual(graph["6"]["inputs"]["first_frame"], ["9", 0])
        self.assertEqual(graph["6"]["inputs"]["last_frame"], ["10", 0])
        for scale, loader in (("9", "7"), ("10", "8")):
            self.assertEqual(graph[scale]["class_type"], "ImageScale")
            self.assertEqual(graph[scale]["inputs"]["image"], [loader, 0])
            self.assertEqual(graph[scale]["inputs"]["crop"], "center")
            self.assertEqual((graph[scale]["inputs"]["width"], graph[scale]["inputs"]["height"]),
                             (graph["6"]["inputs"]["width"], graph["6"]["inputs"]["height"]))

    def test_reference_to_video_wires_refs(self):
        graph = wrapper_workflows.build_minimax_h3_reference_to_video(
            prompt="<Picture 1> <Video 1> <Audio 1>", ref_image_size="max",
            ref_images=["wrapper/i1.png", "wrapper/i2.png", "wrapper/i3.png"],
            ref_videos=["wrapper/v1.mp4"], ref_video_audios=["wrapper/v1.wav"],
            ref_audios=["wrapper/a1.wav", "wrapper/a2.wav"])
        self.assertEqual(graph["6"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(graph["6"]["inputs"]["ref_image_size"], "max")
        # Autogrow slots are addressed by flat dotted ids; a nested dict is
        # dropped by execution (see test_reference_inputs_reach_the_node).
        refs = {k: v for k, v in graph["6"]["inputs"].items() if k.startswith("ref_") and "." in k}
        self.assertEqual(refs, {
            "ref_images.ref_image_0": ["7", 0], "ref_images.ref_image_1": ["8", 0],
            "ref_images.ref_image_2": ["9", 0], "ref_videos.ref_video_0": ["11", 0],
            "ref_video_audios.ref_video_audio_0": ["12", 0],
            "ref_audios.ref_audio_0": ["13", 0], "ref_audios.ref_audio_1": ["14", 0]})
        self.assertNotIn("ref_images", graph["6"]["inputs"])
        self.assertEqual(graph["7"]["class_type"], "LoadImage")
        self.assertEqual(graph["10"]["class_type"], "LoadVideo")
        # the slot takes frames, so the video goes through GetVideoComponents
        self.assertEqual(graph["11"], {"class_type": "GetVideoComponents", "inputs": {"video": ["10", 0]}})
        self.assertEqual(graph["12"]["class_type"], "LoadAudio")  # video soundtrack
        self.assertEqual(graph["13"]["class_type"], "LoadAudio")
        # single-ref uploads (max=1) arrive as a bare filename string
        graph3 = wrapper_workflows.build_minimax_h3_reference_to_video(
            prompt="x", ref_videos="wrapper/v1.mp4", ref_video_audios="wrapper/v1.wav")
        self.assertEqual(graph3["6"]["inputs"]["ref_videos.ref_video_0"], ["8", 0])
        self.assertEqual(graph3["6"]["inputs"]["ref_video_audios.ref_video_audio_0"], ["9", 0])
        # no refs -> no loader nodes, tail starts at 7
        graph2 = wrapper_workflows.build_minimax_h3_reference_to_video(prompt="x")
        self.assertEqual(graph2["7"]["class_type"], "BasicGuider")
        self.assertFalse([k for k in graph2["6"]["inputs"] if k.startswith("ref_") and k != "ref_image_size"])

    def test_decode_nodes_use_the_correct_vae(self):
        # The joint AV latent is a NestedTensor (video [B,24,T,H,W], audio
        # [B,32,2,T]). VAEDecode must unbind the video part -> the VIDEO vae,
        # VAEDecodeAudio the audio part -> the AUDIO vae; swapping them makes
        # the video VAE's 5D memory estimator crash with "tuple index out of
        # range" on the 4D audio latent (sd.py estimate_decode_memory).
        for task_name in ("text", "image"):
            task = wrapper_workflows.WORKFLOWS["minimaxh3"]["tasks"][task_name]
            graph = task["build"](
                prompt="x",
                **({"first_frame": "wrapper/t1.png"} if task_name == "image" else {}),
                seed=1, steps=20, width=672, height=384,
                duration=5.0, scheduler="beta", filename_prefix="wrapper/minimaxh3")
            vae_loaders = {n["inputs"]["vae_name"]: nid for nid, n in graph.items()
                           if n["class_type"] == "VAELoader"}
            decode = {n["class_type"]: n for n in graph.values()}
            self.assertEqual(decode["VAEDecode"]["inputs"]["vae"][0],
                             vae_loaders[wrapper_workflows.MINIMAX_H3_VIDEO_VAE])
            self.assertEqual(decode["VAEDecodeAudio"]["inputs"]["vae"][0],
                             vae_loaders[wrapper_workflows.MINIMAX_H3_AUDIO_VAE])
            # both decoders read the same sampler output (the joint AV latent)
            sampler_out = decode["VAEDecode"]["inputs"]["samples"]
            self.assertEqual(decode["VAEDecodeAudio"]["inputs"]["samples"], sampler_out)

    def test_model_pick_prefers_smallest_present_clip(self):
        w = wrapper_workflows
        disk = set()

        def is_present(folder, filename):
            return filename in disk

        # nvfp4 CLIP on disk -> never touches bf16 even though fp8 is requested,
        # and the preferred encoder needs no note
        disk = {w.MINIMAX_H3_UNET_FP8, w.MINIMAX_H3_CLIP_NVFP4}
        unet, clip, note = w.minimax_h3_model_pick("fp8", ref2va=False, is_present=is_present)
        self.assertEqual(unet, w.MINIMAX_H3_UNET_FP8)
        self.assertEqual(clip, w.MINIMAX_H3_CLIP_NVFP4)
        self.assertIsNone(note)

        # bf16 CLIP is the only encoder present (old downloads) -> used, and the
        # note calls out that the smaller preferred encoder is missing
        disk = {w.MINIMAX_H3_UNET_FP8, w.MINIMAX_H3_CLIP}
        unet, clip, note = w.minimax_h3_model_pick("fp8", ref2va=False, is_present=is_present)
        self.assertEqual(clip, w.MINIMAX_H3_CLIP)
        self.assertIn(w.MINIMAX_H3_CLIP_NVFP4, note)

        # requested UNET missing -> first present variant (fp8) used with note
        disk = {w.MINIMAX_H3_REF2VA_UNET_FP8, w.MINIMAX_H3_CLIP_NVFP4}
        unet, clip, note = w.minimax_h3_model_pick("bf16", ref2va=True, is_present=is_present)
        self.assertEqual(unet, w.MINIMAX_H3_REF2VA_UNET_FP8)
        self.assertEqual(clip, w.MINIMAX_H3_CLIP_NVFP4)
        self.assertIn("not on disk", note)

        # nothing on disk -> requested (or fp8) UNET + nvfp4 CLIP are downloaded
        disk = set()
        unet, clip, note = w.minimax_h3_model_pick("bf16", ref2va=False, is_present=is_present)
        self.assertEqual(unet, w.MINIMAX_H3_UNET_BF16)
        self.assertEqual(clip, w.MINIMAX_H3_CLIP_NVFP4)
        # no explicit quantization -> the templates' int8_convrot + nvfp4 pair
        unet, clip, _ = w.minimax_h3_model_pick(None, ref2va=False, is_present=is_present)
        self.assertEqual(unet, w.MINIMAX_H3_UNET_INT8)
        self.assertEqual(clip, w.MINIMAX_H3_CLIP_NVFP4)
        with self.assertRaises(ValueError):
            w.minimax_h3_model_pick("fp16", False, is_present)

        # models() honors explicit overrides for the download list
        models = w.minimax_h3_models("fp8", ref2va=True, unet_name="x.safetensors", clip_name="y.safetensors")
        self.assertEqual(models[0]["filename"], "x.safetensors")
        self.assertEqual(models[1]["filename"], "y.safetensors")

    def test_models_and_length_helper(self):
        models = wrapper_workflows.minimax_h3_models("int8", ref2va=True)
        self.assertEqual(models[0]["filename"], wrapper_workflows.MINIMAX_H3_REF2VA_UNET_INT8)
        self.assertEqual([m["filename"] for m in models],
                         [wrapper_workflows.MINIMAX_H3_REF2VA_UNET_INT8,
                          wrapper_workflows.MINIMAX_H3_CLIP_INT8,
                          wrapper_workflows.MINIMAX_H3_VIDEO_VAE,
                          wrapper_workflows.MINIMAX_H3_AUDIO_VAE])
        self.assertEqual(wrapper_workflows.minimax_h3_length(5.0), 120)
        self.assertEqual(wrapper_workflows.minimax_h3_length(0.1), 5)  # min 5

    def test_nvfp4_quantization_uses_nvfp4_clip_with_fp8_unet(self):
        # No nvfp4 UNET exists; nvfp4 is a CLIP-only option that drops the text
        # encoder from 66 GB (bf16) / 34 GB (int8) to 16 GB, letting the whole
        # pipeline fit on a 32 GB GPU.
        for ref2va in (False, True):
            models = wrapper_workflows.minimax_h3_models("nvfp4", ref2va=ref2va)
            unet = wrapper_workflows.MINIMAX_H3_REF2VA_UNET_FP8 if ref2va else wrapper_workflows.MINIMAX_H3_UNET_FP8
            self.assertEqual(models[0]["filename"], unet)
            self.assertIn(wrapper_workflows.MINIMAX_H3_CLIP_NVFP4, [m["filename"] for m in models])
            self.assertNotIn(wrapper_workflows.MINIMAX_H3_CLIP, [m["filename"] for m in models])
        self.assertTrue(all(
            q in wrapper_workflows.MINIMAX_H3_QUANT_MODELS for q in ("fp8", "int8", "bf16", "nvfp4")))

    def test_fit_canvas_rounds_up_and_caps_area(self):
        w = wrapper_workflows
        # already on the grid and inside the cap -> untouched, nothing to report
        self.assertEqual(w.minimax_h3_fit_canvas(1344, 768), (1344, 768, None))
        # off-grid rounds up, never below what was asked for
        self.assertEqual(w.minimax_h3_fit_canvas(720, 720)[:2], (736, 736))
        # over the native canvas -> scaled down, aspect kept, note explains it
        for width, height in ((1920, 1080), (3840, 2160), (1080, 1920), (4000, 1000)):
            fitted_w, fitted_h, note = w.minimax_h3_fit_canvas(width, height)
            self.assertEqual((fitted_w % 32, fitted_h % 32), (0, 0))
            self.assertLessEqual(fitted_w * fitted_h, w.MINIMAX_H3_MAX_PIXELS)
            self.assertIn("native canvas", note)
            self.assertAlmostEqual(fitted_w / fitted_h, width / height, delta=0.15)

    def test_budget_matches_measured_envelope(self):
        """The guard has to agree with what the reference box actually did:
        letting an over-budget job through OOM-kills the whole server."""
        w = wrapper_workflows
        frames = lambda seconds: w.minimax_h3_align_frames(w.minimax_h3_length(seconds))
        self.assertEqual(frames(5), 124)  # 5s snaps up to the 17k+5 grid

        for width, height, seconds in ((864, 480, 5), (1152, 640, 5), (736, 736, 5)):
            self.assertIsNone(w.minimax_h3_check_budget(width, height, frames(seconds)),
                              f"{width}x{height} {seconds}s completed on the reference box")
        for width, height, seconds in ((1344, 768, 5), (960, 544, 10), (1344, 768, 8),
                                       (1344, 768, 10)):
            self.assertIsNotNone(w.minimax_h3_check_budget(width, height, frames(seconds)),
                                 f"{width}x{height} {seconds}s OOM-killed the reference box")

        # the form defaults must be inside the budget, or every default request 400s
        self.assertIsNone(w.minimax_h3_check_budget(
            w.MINIMAX_H3_DEFAULT_WIDTH, w.MINIMAX_H3_DEFAULT_HEIGHT, w.MINIMAX_H3_MAX_FRAMES))
        # and the advertised maximum duration must not itself be over the cap
        self.assertLessEqual(frames(w.MINIMAX_H3_MAX_DURATION), w.MINIMAX_H3_MAX_FRAMES)

    def test_over_length_is_rejected_at_any_resolution(self):
        """Frame count is the hard limit -- shrinking the canvas must not buy
        back a longer clip, because the video VAE does not tile temporally."""
        w = wrapper_workflows
        too_many = w.MINIMAX_H3_MAX_FRAMES + 17
        for width, height in ((1344, 768), (864, 480), (64, 64)):
            note = w.minimax_h3_check_budget(width, height, too_many)
            self.assertIsNotNone(note)
            self.assertIn("frames", note)


class TestIdeogram4Graph(unittest.TestCase):
    def test_structure(self):
        graph = wrapper_workflows.build_ideogram4_text2img(prompt="test", width=1000, height=700, seed=42)
        self.assertEqual(len(graph), 15)
        class_types = [node["class_type"] for node in graph.values()]
        for expected in ("UNETLoader", "CLIPLoader", "VAELoader", "CLIPTextEncode",
                         "ConditioningZeroOut", "CFGOverride", "DualModelGuider",
                         "Ideogram4Scheduler", "RandomNoise", "KSamplerSelect",
                         "EmptyFlux2LatentImage", "SamplerCustomAdvanced", "VAEDecode", "SaveImage"):
            self.assertIn(expected, class_types)
        # link integrity: every ["id", slot] reference exists
        ids = set(graph)
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], ids)

    def test_size_rounding_and_presets(self):
        graph = wrapper_workflows.build_ideogram4_text2img(
            prompt="test", width=1000, height=700, steps=12, mu=0.5, std=1.75)
        scheduler = graph["9"]["inputs"]
        self.assertEqual(scheduler["width"], 1008)  # (1000+15)//16*16
        self.assertEqual(scheduler["height"], 704)  # (700+15)//16*16
        self.assertEqual(scheduler["steps"], 12)
        self.assertEqual(scheduler["mu"], 0.5)
        latent = graph["12"]["inputs"]
        self.assertEqual((latent["width"], latent["height"]), (1008, 704))

    def test_minimum_size(self):
        graph = wrapper_workflows.build_ideogram4_text2img(prompt="test", width=10, height=10)
        self.assertEqual(graph["9"]["inputs"]["width"], 256)
        self.assertEqual(graph["9"]["inputs"]["height"], 256)

    def test_model_names(self):
        graph = wrapper_workflows.build_ideogram4_text2img(prompt="test")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], wrapper_workflows.IDEOGRAM4_UNET)
        self.assertEqual(graph["2"]["inputs"]["unet_name"], wrapper_workflows.IDEOGRAM4_UNET_UNCONDITIONAL)
        self.assertEqual(graph["3"]["inputs"]["clip_name"], wrapper_workflows.IDEOGRAM4_CLIP)
        self.assertEqual(graph["4"]["inputs"]["vae_name"], wrapper_workflows.IDEOGRAM4_VAE)


try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _prepare_env_for_comfy_import():
    """Allow comfy to import on environments whose torch predates the
    comfy-kitchen requirement (torch < 2.4 lacks torch.library.custom_op).

    A stub module is enough: comfy.quant_ops guards the tensor sub-import with
    `except ImportError` and degrades to eager ops, so only the initial
    `import comfy_kitchen` must succeed.
    """
    try:
        import comfy_kitchen  # noqa: F401
    except Exception:
        import sys
        import types

        stub = types.ModuleType("comfy_kitchen")
        stub.registry = types.SimpleNamespace(
            disable=lambda *args, **kwargs: None,
            enable=lambda *args, **kwargs: None,
        )
        stub.list_backends = lambda: {}
        sys.modules["comfy_kitchen"] = stub

    # torch.serialization.add_safe_globals only exists on torch >= 2.4; the
    # test never loads checkpoints, so a no-op preserves the import path.
    if not hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals = lambda *args, **kwargs: None


@unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
class TestGraphValidation(unittest.TestCase):
    TEST_IMAGE = "wrapper/test_input.png"
    # 1x1 transparent PNG
    TEST_IMAGE_BYTES = (
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    def setUp(self):
        import folder_paths

        image_path = os.path.join(folder_paths.get_input_directory(), "wrapper", "test_input.png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(self.TEST_IMAGE_BYTES)

    def tearDown(self):
        import folder_paths

        image_path = os.path.join(folder_paths.get_input_directory(), "wrapper", "test_input.png")
        if os.path.isfile(image_path):
            os.remove(image_path)

    def test_graph_passes_core_validation(self):
        _prepare_env_for_comfy_import()
        import nodes  # noqa: F401
        import api_wrapper.routes  # noqa: F401  (imports cleanly in a real env)

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        import execution

        graph = _sample_graph()
        graph["4"]["inputs"]["image"] = self.TEST_IMAGE
        valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt(
            "3f4e2c1a-9b8d-4a5e-8c6f-1d2e3f4a5b6c", graph, None))
        self.assertTrue(valid, f"graph rejected: {error}\nnode_errors: {node_errors}")
        self.assertTrue(outputs)

    def test_ideogram4_graph_passes_core_validation(self):
        _prepare_env_for_comfy_import()
        import nodes  # noqa: F401
        import api_wrapper.routes  # noqa: F401

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        import execution

        graph = wrapper_workflows.build_ideogram4_text2img(prompt="a test prompt")
        valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt(
            "7f9e3c2b-1a4d-4b5e-9c6f-2e3f4a5b6c7d", graph, None))
        self.assertTrue(valid, f"ideogram4 graph rejected: {error}\nnode_errors: {node_errors}")
        self.assertTrue(outputs)

    def test_klein_txt2img_graph_passes_core_validation(self):
        _prepare_env_for_comfy_import()
        import nodes  # noqa: F401
        import api_wrapper.routes  # noqa: F401

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        import execution

        graph = wrapper_workflows.build_flux2_klein_9b_text2img(prompt="a test prompt")
        valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt(
            "8a0f4d3c-2b5e-4c6f-8d70-3f4a5b6c7d8e", graph, None))
        self.assertTrue(valid, f"klein txt2img graph rejected: {error}\nnode_errors: {node_errors}")
        self.assertTrue(outputs)


    def test_minimax_h3_graphs_pass_core_validation(self):
        _prepare_env_for_comfy_import()
        import nodes  # noqa: F401
        import api_wrapper.routes  # noqa: F401
        import folder_paths

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        # The loader combos only accept files that exist; dummy placeholders
        # satisfy the graph validation without downloading the real models.
        placeholder_paths = []
        for folder, name in (
            ("diffusion_models", wrapper_workflows.MINIMAX_H3_UNET_INT8),
            ("diffusion_models", wrapper_workflows.MINIMAX_H3_REF2VA_UNET_INT8),
            ("text_encoders", wrapper_workflows.MINIMAX_H3_CLIP_NVFP4),
            ("vae", wrapper_workflows.MINIMAX_H3_VIDEO_VAE),
            ("vae", wrapper_workflows.MINIMAX_H3_AUDIO_VAE),
        ):
            path = os.path.join(folder_paths.get_folder_paths(folder)[0], name)
            with open(path, "wb") as f:
                f.write(b"placeholder")
            placeholder_paths.append(path)
        for name in ("test_input.mp4", "test_input.wav"):
            path = os.path.join(folder_paths.get_input_directory(), "wrapper", name)
            with open(path, "wb") as f:
                f.write(b"placeholder")
            placeholder_paths.append(path)
        try:
            import execution

            for label, graph in (
                ("t2v", wrapper_workflows.build_minimax_h3_text_to_video(prompt="a test prompt")),
                ("i2v", wrapper_workflows.build_minimax_h3_image_to_video(
                    prompt="a test prompt", first_frame="wrapper/test_input.png")),
                ("ref2va", wrapper_workflows.build_minimax_h3_reference_to_video(
                    prompt="a test prompt",
                    ref_images=["wrapper/test_input.png"],
                    ref_videos=["wrapper/test_input.mp4"],
                    ref_audios=["wrapper/test_input.wav"])),
            ):
                valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt(
                    f"9b1a5e4d-{abs(hash(label)) % 100000:05d}-4d7e-8f90-4a5b6c7d8e9f", graph, None))
                self.assertTrue(valid, f"minimax {label} graph rejected: {error}\nnode_errors: {node_errors}")
                self.assertTrue(outputs, f"minimax {label} produced no output nodes")
        finally:
            for path in placeholder_paths:
                os.remove(path)

    def test_reference_inputs_reach_the_node(self):
        """Validation passes whether or not the references are wired, so this
        runs the resolution execution does before calling the node: what the
        node receives as ref_images / ref_audios. A nested dict used to come
        through as {} — every Ref2VA render was text-only."""
        _prepare_env_for_comfy_import()
        import nodes
        from comfy_api.latest import _io

        async def _init():
            await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)

        asyncio.run(_init())

        import execution

        graph = wrapper_workflows.build_minimax_h3_reference_to_video(
            prompt="<Picture 1> <Picture 2> <Audio 1>",
            ref_images=["wrapper/a.png", "wrapper/b.png"], ref_audios=["wrapper/c.wav"])
        node_class = nodes.NODE_CLASS_MAPPINGS[graph["6"]["class_type"]]
        received, _missing, v3_data = execution.get_input_data(graph["6"]["inputs"], node_class, "6")
        received = _io.build_nested_inputs(received, v3_data)
        self.assertEqual(sorted(received["ref_images"]), ["ref_image_0", "ref_image_1"])
        self.assertEqual(sorted(received["ref_audios"]), ["ref_audio_0"])
        self.assertEqual(received.get("ref_videos"), {})


if __name__ == "__main__":
    unittest.main()

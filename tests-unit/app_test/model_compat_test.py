from app.model_compat import describe, judge, precision

GIB = 1024 ** 3
# Fixed example hardware so the levels do not depend on the machine running the tests.
FAKE_HW = {
    "type": "cuda",
    "vram_total": 16 * GIB,
    "ram_total": 32 * GIB,
    "supports": {"bf16": True, "fp8": True, "fp4": True},
}
FAKE_OLD_GPU = {**FAKE_HW, "supports": {"bf16": False, "fp8": False, "fp4": False}}


def test_precision_from_file_name():
    assert precision("z_image_turbo_nvfp4.safetensors") == "FP4"
    assert precision("flux1-dev-fp8-e4m3fn.safetensors") == "FP8"
    assert precision("boogu_image_edit_int8_convrot.safetensors") == "INT8"
    assert precision("lumina_2_model_bf16.safetensors") == "BF16"
    assert precision("flux1-dev-Q4_K_S.gguf") == "GGUF Q4_K_S"
    assert precision("qwen-image-IQ4_XS.gguf") == "GGUF IQ4_XS"
    assert precision("ae.safetensors") is None


def test_judge_levels():
    assert judge(11 * GIB, "BF16", FAKE_HW)["level"] == "fits"
    assert judge(15 * GIB, "BF16", FAKE_HW)["level"] == "offload"
    assert judge(40 * GIB, "BF16", FAKE_HW)["level"] == "no"


def test_judge_notes_unsupported_precision():
    notes = judge(4 * GIB, "FP4", FAKE_OLD_GPU)["notes"]
    assert any("FP4" in note for note in notes)


def test_judge_without_gpu_never_fits():
    cpu = {**FAKE_HW, "type": "cpu"}
    assert judge(1 * GIB, None, cpu)["level"] == "offload"


def test_describe_uses_folder_and_file_name():
    info = describe("diffusion_models", "flux/flux1-dev-fp8.safetensors", 12 * GIB, FAKE_HW)
    assert info["path"] == "diffusion_models/flux/flux1-dev-fp8.safetensors"
    assert info["name"] == "flux1-dev-fp8.safetensors"
    assert info["precision"] == "FP8"
    assert info["level"] == "fits"

import torch

from comfy.cli_args import args as cli_args

_original_cli_args_cpu = cli_args.cpu
try:
    if not torch.cuda.is_available():
        cli_args.cpu = True

    import comfy.sd
    import comfy.supported_models
finally:
    cli_args.cpu = _original_cli_args_cpu


def test_minimax_music3_fp16_manual_cast_only_for_bf16_device(monkeypatch):
    bf16_device = object()
    fp16_device = object()

    monkeypatch.setattr(
        comfy.supported_models.comfy.model_management,
        "should_use_bf16",
        lambda device=None: device is bf16_device,
    )

    bf16_config = comfy.supported_models.MiniMaxMusic3({"audio_model": "minimax_music3"})
    bf16_config.set_inference_dtype(torch.float16, None, device=bf16_device)
    assert bf16_config.manual_cast_dtype is torch.bfloat16

    fp16_config = comfy.supported_models.MiniMaxMusic3({"audio_model": "minimax_music3"})
    fp16_config.set_inference_dtype(torch.float16, None, device=fp16_device)
    assert fp16_config.manual_cast_dtype is None

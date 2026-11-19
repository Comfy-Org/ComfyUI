import torch

from comfy.cli_args import args

args.cpu = True

import comfy.sd
from comfy.text_encoders.sensenova import SenseNovaTextEncoder, SenseNovaTokenizer
from comfy.supported_models_base import ClipTarget
from comfy_extras.nodes_sensenova import SenseNovaConditioning


class PatchEmbedding(torch.nn.Conv2d):
    def forward(self, image):
        return super().forward(image).flatten(2).transpose(1, 2)


def _run_prepared_generation(*, generation_mode, clip, conditioning, max_think_tokens, negative=None, samples=None):
    """Exercise text decoding with explicitly constructed token prefixes."""
    if generation_mode == "interleave":
        return SenseNovaConditioning._execute_interleave(
            clip=clip, positive=conditioning, negative=negative,
            samples=samples, max_text_tokens=max_think_tokens,
        )
    return SenseNovaConditioning._execute_image(
        clip=clip, conditioning=conditioning, max_tokens=max_think_tokens,
    )


def _generation_input_ids():
    return torch.tensor(
        [[151644, 8948, 198, 1, 151645, 198,
          151644, 872, 198, 2, 151645, 198,
          151644, 77091, 198, 151670]],
        dtype=torch.long,
    )


def _tokenize_generation_prompt(text):
    tokenizer = SenseNovaTokenizer()
    values = tokenizer.tokenize_with_weights(text)["sensenova_u15"][0]
    return torch.tensor([[int(value[0]) for value in values]], dtype=torch.long)


def _text_clip(token_ids):
    """A small deterministic language model using the real token decode path."""
    width = len(token_ids)
    embeddings = torch.zeros(151936, width)
    head = torch.zeros_like(embeddings)
    for index, (previous, following) in enumerate(zip([1, *token_ids], token_ids)):
        embeddings[previous, index] = 1
        head[following, index] = 1
    backbone = torch.nn.Module()
    backbone.embed_tokens = torch.nn.Embedding.from_pretrained(embeddings)
    backbone.layers = torch.nn.ModuleList()
    backbone.norm = torch.nn.Identity()
    language_model = torch.nn.Module()
    language_model.model = backbone
    language_model.lm_head = torch.nn.Linear(width, 151936, bias=False)
    language_model.lm_head.weight = torch.nn.Parameter(head, requires_grad=False)
    encoder = object.__new__(SenseNovaTextEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.has_lm_head = True
    encoder.language_model = language_model
    encoder.dtypes = {torch.float32}
    encoder.execution_device = torch.device("cpu")
    target = ClipTarget(SenseNovaTokenizer, lambda **kwargs: encoder)
    return comfy.sd.CLIP(target, model_options={"load_device": torch.device("cpu")}, disable_dynamic=True)


def prefix_arguments(metadata, device, dtype):
    encoder = object.__new__(SenseNovaTextEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.execution_device = device
    args = encoder._prefix_arguments(metadata, dtype)
    return tuple(args[key] for key in ("text_input_ids", "reference_images", "prefix_indexes", "prefix_mask"))


def prepared_prefix(**metadata):
    encoder = object.__new__(SenseNovaTextEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.execution_device = metadata.pop("device", torch.device("cpu"))
    dtype = metadata.pop("dtype", torch.float32)
    metadata["image_only"] = metadata.pop("prompt_type", None) == "negative"
    return {key: value for key, value in encoder._prefix_arguments(metadata, dtype).items() if value is not None}

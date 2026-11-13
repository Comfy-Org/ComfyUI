import torch

from comfy.cli_args import args

args.cpu = True

from comfy import model_base
import comfy.model_patcher
from comfy.ldm.sensenova import model as sensenova_model
from comfy.text_encoders.sensenova import SenseNovaTokenizer
from comfy_extras.nodes_sensenova import TextGenerateSenseNova


class PatchEmbedding(torch.nn.Conv2d):
    def forward(self, image):
        return super().forward(image).flatten(2).transpose(1, 2)


def _run_prepared_generation(*, generation_mode, model, clip, conditioning, max_think_tokens, negative=None, samples=None):
    """Exercise text decoding with explicitly constructed token prefixes."""
    if generation_mode == "interleave":
        return TextGenerateSenseNova._execute_interleave(
            model=model, clip=clip, positive=conditioning, negative=negative,
            samples=samples, max_text_tokens=max_think_tokens,
        )
    return TextGenerateSenseNova._execute_image(
        model=model, clip=clip, conditioning=conditioning, max_tokens=max_think_tokens,
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


def _text_model_patcher(token_ids):
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
    diffusion_model = object.__new__(sensenova_model.SenseNovaU15)
    torch.nn.Module.__init__(diffusion_model)
    diffusion_model.has_lm_head = True
    diffusion_model.dtype = torch.float32
    diffusion_model.language_model = language_model
    model = object.__new__(model_base.SenseNovaU15)
    torch.nn.Module.__init__(model)
    model.diffusion_model = diffusion_model
    model.manual_cast_dtype = None
    model.current_patcher = None
    return comfy.model_patcher.ModelPatcher(
        model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )

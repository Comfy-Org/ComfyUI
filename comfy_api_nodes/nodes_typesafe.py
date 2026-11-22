import uuid

from typing_extensions import override

from comfy.utils import common_upscale
from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.typesafe import (
    TypeSafeAnswer,
    TypeSafeChoiceQuestion,
    TypeSafeImageAsset,
    TypeSafeNoulQuestion,
    TypeSafeScoreQuestion,
    TypeSafeSystemOneRequest,
    TypeSafeSystemOneResponse,
)
from comfy_api_nodes.util import (
    get_number_of_images,
    router_sync_op,
    tensor_to_base64_string,
    validate_string,
)
from server import PromptServer

JEV_MAX_IMAGES = 4
JEV_PICK_MAX_IMAGES = 32
JEV_PICK_IMAGE_SIDES = ((4, 1568), (8, 1024), (16, 768), (32, 512))
JEV_MIN_IMAGE_SIDE = 280
JEV_MAX_IMAGE_SIDE = 1568
JEV_PRICE_BADGE = IO.PriceBadge(
    expr="""{"type":"usd","usd":0.06006,"format":{"approximate":true,"suffix":" per 1M tokens"}}""",
)


def _image_to_asset(image: Input.Image, max_side: int) -> TypeSafeImageAsset:
    image = image[..., :3]
    height, width = image.shape[1], image.shape[2]
    scale = min(1.0, max_side / max(width, height))
    new_width = max(JEV_MIN_IMAGE_SIDE, round(width * scale))
    new_height = max(JEV_MIN_IMAGE_SIDE, round(height * scale))
    if (new_width, new_height) != (width, height):
        image = common_upscale(image.movedim(-1, 1), new_width, new_height, "lanczos", "disabled").movedim(1, -1)
    return TypeSafeImageAsset(data_base64=tensor_to_base64_string(image, total_pixels=None, mime_type="image/jpeg"))


def _jev_inputs() -> list[Input]:
    return [
        IO.Autogrow.Input(
            "images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("image"),
                names=[f"image_{i}" for i in range(1, JEV_MAX_IMAGES + 1)],
                min=0,
            ),
            optional=True,
            tooltip=f"The images the question is about. Up to {JEV_MAX_IMAGES} images in total.",
        ),
        IO.String.Input(
            "text",
            multiline=True,
            force_input=True,
            optional=True,
            tooltip="The text the question is about, such as a prompt or an LLM response.",
        ),
    ]


def _jev_state(
    text: str,
    images: dict[str, Input.Image],
    max_side: int,
) -> tuple[dict[str, str | list[str]], dict[str, TypeSafeImageAsset]]:
    state: dict[str, str | list[str]] = {}
    if text.strip():
        state["text"] = text.strip()
    assets: dict[str, TypeSafeImageAsset] = {}
    for name, image in images.items():
        refs = []
        for i in range(get_number_of_images(image)):
            ref = str(uuid.uuid4())
            assets[ref] = _image_to_asset(image[i : i + 1], max_side)
            refs.append(ref)
        state[name] = refs[0] if len(refs) == 1 else refs
    if not state:
        raise ValueError("Connect text or an image for the question to be about.")
    return state, assets


def _connected_images(images: dict[str, Input.Image]) -> dict[str, Input.Image]:
    connected = {name: image for name, image in images.items() if image is not None and get_number_of_images(image)}
    count = sum(get_number_of_images(image) for image in connected.values())
    if count > JEV_MAX_IMAGES:
        raise ValueError(f"Up to {JEV_MAX_IMAGES} images are supported in total, got {count}.")
    return connected


async def _run_jev(
    cls: type[IO.ComfyNode],
    model: str,
    question: TypeSafeNoulQuestion | TypeSafeChoiceQuestion | TypeSafeScoreQuestion,
    state: dict[str, str | list[str]],
    assets: dict[str, TypeSafeImageAsset],
) -> TypeSafeAnswer:
    response = await router_sync_op(
        cls,
        f"typesafe/{model}",
        response_model=TypeSafeSystemOneResponse,
        data=TypeSafeSystemOneRequest(model=model, state=state, questions={"answer": question}, assets=assets or None),
    )
    return response.answers["answer"]


def _parse_options(options: str) -> list[str]:
    names: list[str] = []
    for number, line in enumerate(options.splitlines(), 1):
        name = line.strip()
        if not name:
            continue
        if len(name) > 40:
            raise ValueError(f"The option on line {number} is longer than 40 characters.")
        if list(map(str.casefold, name)) in (list(map(str.casefold, n)) for n in names):
            raise ValueError(f"The option '{name}' is listed twice. Options must differ by more than letter case.")
        names.append(name)
    if not 2 <= len(names) <= 50:
        raise ValueError(f"Give 2 to 50 options, one per line. Got {len(names)}.")
    return names


class TypeSafeJevYesNo(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TypeSafeJevYesNo",
            display_name="TypeSafe Jev (Yes/No)",
            category="partner/text/TypeSafe",
            description="Ask a yes/no question about text or images. "
            "Connect `result` to an If/Else Switch node to choose between two paths.",
            search_aliases=["if", "condition", "check", "filter", "classify", "verify", "yes no", "boolean"],
            inputs=[
                IO.Combo.Input("model", options=["jev-1.14.0"]),
                IO.String.Input(
                    "question",
                    multiline=True,
                    default="Is there a person in the image?",
                    tooltip="A question with a yes or no answer about what is connected. "
                    "Ask it so that yes is the case you act on. With several inputs, use the socket names, "
                    "for example 'Does image_2 show the same person as image_1?'",
                ),
                IO.Float.Input(
                    "threshold",
                    default=0.5,
                    min=0.01,
                    max=0.99,
                    step=0.01,
                    tooltip="The answer is yes when the probability of yes is at least this value.",
                ),
                *_jev_inputs(),
            ],
            outputs=[
                IO.Boolean.Output(display_name="result"),
                IO.String.Output(display_name="answer"),
                IO.Float.Output(display_name="probability"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=JEV_PRICE_BADGE,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        question: str,
        threshold: float,
        images: IO.Autogrow.Type,
        text: str = "",
    ) -> IO.NodeOutput:
        validate_string(question, field_name="question", min_length=1, max_length=1000)
        validate_string(text, field_name="text", max_length=3000)
        state, assets = _jev_state(text, _connected_images(images), JEV_MAX_IMAGE_SIDE)
        answer = await _run_jev(cls, model, TypeSafeNoulQuestion(instructions=question.strip()), state, assets)
        result = answer.noul >= threshold
        verdict = "yes" if result else "no"
        PromptServer.instance.send_progress_text(f"{verdict} · {answer.noul:.2f}", cls.hidden.unique_id)
        return IO.NodeOutput(result, verdict, answer.noul)


class TypeSafeJevChoice(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TypeSafeJevChoice",
            display_name="TypeSafe Jev (Choice)",
            category="partner/text/TypeSafe",
            description="Pick one of your options for text or images. "
            "Connect `choice` to a Branch node whose inputs carry the option names, or use `option_index`.",
            search_aliases=["classify", "category", "route", "select", "label", "choose"],
            inputs=[
                IO.Combo.Input("model", options=["jev-1.14.0"]),
                IO.String.Input(
                    "question",
                    multiline=True,
                    default="What style is the image?",
                    tooltip="The question the options answer. With several inputs, use the socket names, "
                    "for example 'Which season is shown in image_1?'",
                ),
                IO.String.Input(
                    "options",
                    multiline=True,
                    default="photo\nillustration\n3d render\nanime\nsketch\nother",
                    tooltip="One option per line. "
                    "Add an option such as `other` for inputs that fit none of them.",
                ),
                *_jev_inputs(),
            ],
            outputs=[
                IO.String.Output(display_name="choice"),
                IO.Int.Output(
                    display_name="option_index",
                    tooltip="Position of the chosen option in the options list, starting at 0.",
                ),
                IO.Float.Output(
                    display_name="confidence",
                    tooltip="How sure Jev is, from 0 (no clear winner) to 1 (certain).",
                ),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=JEV_PRICE_BADGE,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        question: str,
        options: str,
        images: IO.Autogrow.Type,
        text: str = "",
    ) -> IO.NodeOutput:
        validate_string(question, field_name="question", min_length=1, max_length=1000)
        validate_string(text, field_name="text", max_length=3000)
        names = _parse_options(options)
        state, assets = _jev_state(text, _connected_images(images), JEV_MAX_IMAGE_SIDE)
        answer = await _run_jev(
            cls,
            model,
            TypeSafeChoiceQuestion(instructions=question.strip(), criteria={name: "" for name in names}),
            state,
            assets,
        )
        PromptServer.instance.send_progress_text(f"{answer.choice} · {answer.confidence:.2f}", cls.hidden.unique_id)
        return IO.NodeOutput(answer.choice, names.index(answer.choice), answer.confidence)


class TypeSafeJevPickImage(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TypeSafeJevPickImage",
            display_name="TypeSafe Jev (Pick Image)",
            category="partner/text/TypeSafe",
            description="Pick one image from a batch. Connect `index` to Get Image from Batch to keep the picked image.",
            search_aliases=["pick best", "best image", "select image", "best of", "rank images"],
            inputs=[
                IO.Combo.Input("model", options=["jev-1.14.0"]),
                IO.String.Input(
                    "question",
                    multiline=True,
                    default="Which image best matches the prompt in text? "
                    "If there is no text, which image has the best quality?",
                    tooltip="Which image to pick. Connect the prompt to `text` to pick the image that matches it best.",
                ),
                IO.Image.Input("images", tooltip=f"The images to choose from, 2 to {JEV_PICK_MAX_IMAGES}."),
                IO.String.Input(
                    "text",
                    multiline=True,
                    force_input=True,
                    optional=True,
                    tooltip="The text the question is about, such as the prompt the images were made from.",
                ),
            ],
            outputs=[
                IO.Int.Output(
                    display_name="index",
                    tooltip="Position of the picked image in the batch, starting at 0.",
                ),
                IO.Float.Output(
                    display_name="confidence",
                    tooltip="How sure Jev is, from 0 (no clear winner) to 1 (certain).",
                ),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=JEV_PRICE_BADGE,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        question: str,
        images: Input.Image,
        text: str = "",
    ) -> IO.NodeOutput:
        validate_string(question, field_name="question", min_length=1, max_length=1000)
        validate_string(text, field_name="text", max_length=3000)
        count = get_number_of_images(images)
        if not 2 <= count <= JEV_PICK_MAX_IMAGES:
            raise ValueError(f"Pick Image takes 2 to {JEV_PICK_MAX_IMAGES} images, got {count}.")
        names = [f"image_{i}" for i in range(1, count + 1)]
        side = next(side for limit, side in JEV_PICK_IMAGE_SIDES if count <= limit)
        state, assets = _jev_state(text, {name: images[i : i + 1] for i, name in enumerate(names)}, side)
        answer = await _run_jev(
            cls,
            model,
            TypeSafeChoiceQuestion(instructions=question.strip(), criteria={name: "" for name in names}),
            state,
            assets,
        )
        index = names.index(answer.choice)
        PromptServer.instance.send_progress_text(
            f"image {index + 1} of {count} · {answer.confidence:.2f}", cls.hidden.unique_id
        )
        return IO.NodeOutput(index, answer.confidence)


class TypeSafeJevScore(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TypeSafeJevScore",
            display_name="TypeSafe Jev (Score)",
            category="partner/text/TypeSafe",
            description="Rate text or images from 1 to 10. Connect `result` to an If/Else Switch node to act on the score.",
            search_aliases=["rate", "rating", "quality", "grade", "judge", "evaluate"],
            inputs=[
                IO.Combo.Input("model", options=["jev-1.14.0"]),
                IO.String.Input(
                    "question",
                    multiline=True,
                    default="Rate the technical quality of the image from 1 (worst) to 10 (best).",
                    tooltip="What to rate, on a scale from 1 to 10. Say what 1 and 10 mean, as in the example.",
                ),
                IO.Float.Input(
                    "threshold",
                    default=5.0,
                    min=1.0,
                    max=10.0,
                    step=0.1,
                    tooltip="`result` is true when the score is at least this value.",
                ),
                *_jev_inputs(),
            ],
            outputs=[
                IO.Boolean.Output(display_name="result"),
                IO.Float.Output(display_name="score"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=JEV_PRICE_BADGE,
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        question: str,
        threshold: float,
        images: IO.Autogrow.Type,
        text: str = "",
    ) -> IO.NodeOutput:
        validate_string(question, field_name="question", min_length=1, max_length=1000)
        validate_string(text, field_name="text", max_length=3000)
        state, assets = _jev_state(text, _connected_images(images), JEV_MAX_IMAGE_SIDE)
        answer = await _run_jev(
            cls,
            model,
            TypeSafeScoreQuestion(instructions=question.strip(), criteria=[str(i) for i in range(1, 11)]),
            state,
            assets,
        )
        score = round(answer.score + 1, 2)
        result = score >= threshold
        PromptServer.instance.send_progress_text(
            f"{score:.1f} / 10 · {'pass' if result else 'fail'}", cls.hidden.unique_id
        )
        return IO.NodeOutput(result, score)


class TypeSafeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TypeSafeJevYesNo,
            TypeSafeJevChoice,
            TypeSafeJevPickImage,
            TypeSafeJevScore,
        ]


async def comfy_entrypoint() -> TypeSafeExtension:
    return TypeSafeExtension()

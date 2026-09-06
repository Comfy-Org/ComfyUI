import torch


IMAGE_CONTEXT_ID = 151669
IMAGE_START_ID = 151670
IMAGE_END_ID = 151671
IM_START_ID = 151644
IM_END_ID = 151645
USER_ID = 872
ASSISTANT_ID = 77091
NEWLINE_ID = 198
IMAGE_LABEL_ID = 1906
HYPHEN_ID = 12
DIGIT_ZERO_ID = 15
COLON_ID = 25


def preprocess_reference(image):
    """Convert a ComfyUI reference image to normalized BCHW RGB."""

    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image[:, :, :, :3].movedim(-1, 1).float()
    if image.shape[1] == 0:
        image = image.new_zeros((image.shape[0], 3, *image.shape[-2:]))
    elif image.shape[1] < 3:
        repeats = (3 + image.shape[1] - 1) // image.shape[1]
        image = image.repeat(1, repeats, 1, 1)[:, :3]
    mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (image - mean) / std


def split_reference_batches(images):
    """Split reference image batches into single-image tensors."""

    references = []
    for image in images:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        references.extend(image[index : index + 1] for index in range(image.shape[0]))
    return references


def preprocess_references(images):
    """Normalize each reference image independently."""

    return [preprocess_reference(image) for image in split_reference_batches(images)]


def _image_tokens(token_height, token_width):
    return (
        [IMAGE_START_ID]
        + [IMAGE_CONTEXT_ID] * (token_height * token_width)
        + [IMAGE_END_ID]
    )


def _image_label_tokens(index):
    digits = (DIGIT_ZERO_ID + int(digit) for digit in str(index + 1))
    return (IMAGE_LABEL_ID, HYPHEN_ID, *digits, COLON_ID)


def conditioned_input_length(
    input_length, reference_grids, image_only=False, append_image_start=True
):
    """Return the token length after inserting reference-image blocks."""

    image_token_count = sum(height * width for height, width in reference_grids)
    if image_only:
        return (
            image_token_count
            + 9
            + 2 * len(reference_grids)
            - int(not append_image_start)
        )
    label_count = (
        sum(len(_image_label_tokens(index)) for index in range(len(reference_grids)))
        if len(reference_grids) > 1
        else 0
    )
    return input_length + image_token_count + 3 * len(reference_grids) + label_count


def condition_input_ids(
    input_ids, reference_grids, image_only=False, append_image_start=True
):
    """Insert reference-image token blocks into a SenseNova prompt."""

    image_blocks = [_image_tokens(height, width) for height, width in reference_grids]
    if image_only:
        values = (
            [IM_START_ID, USER_ID, NEWLINE_ID]
            + [token for block in image_blocks for token in block]
            + [
                IM_END_ID,
                NEWLINE_ID,
                IM_START_ID,
                ASSISTANT_ID,
                NEWLINE_ID,
            ]
            + ([IMAGE_START_ID] if append_image_start else [])
        )
        return torch.tensor([values], dtype=torch.long, device=input_ids.device)

    values = input_ids[0].tolist()
    starts = [index for index, value in enumerate(values) if value == IM_START_ID]
    insert_at = starts[1] + 3 if len(starts) > 1 else len(values)
    inserted = []
    for index, block in enumerate(image_blocks):
        if len(image_blocks) > 1:
            inserted.extend(_image_label_tokens(index))
        inserted.extend(block)
        inserted.append(NEWLINE_ID)
    values[insert_at:insert_at] = inserted
    return torch.tensor([values], dtype=torch.long, device=input_ids.device)


def thw_indexes(input_ids, reference_grids):
    """Build temporal, height, and width positions for prefix tokens."""

    values = input_ids[0]
    image_start_shift = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=values.device),
            (values == IMAGE_START_ID).long(),
        )
    )[:-1]
    not_image = (values != IMAGE_CONTEXT_ID).long()
    time_indexes = (image_start_shift + not_image).cumsum(0) - 1
    height_indexes = torch.zeros_like(time_indexes)
    width_indexes = torch.zeros_like(time_indexes)
    selected = values == IMAGE_CONTEXT_ID
    height_positions = []
    width_positions = []
    for token_height, token_width in reference_grids:
        positions = torch.arange(
            token_height * token_width, dtype=torch.long, device=values.device
        )
        height_positions.append(positions // token_width)
        width_positions.append(positions % token_width)
    if height_positions:
        height_indexes[selected] = torch.cat(height_positions)
        width_indexes[selected] = torch.cat(width_positions)
    return torch.stack((time_indexes, height_indexes, width_indexes)).unsqueeze(0)


def block_causal_mask(time_indexes, dtype=torch.float32):
    """Build the block-causal attention mask used by the prefix decoder."""

    values = time_indexes[0, 0]
    length = values.shape[0]
    same_block = values[:, None] == values[None, :]
    positions = torch.arange(length, device=values.device)
    causal = positions[None, :] <= positions[:, None]
    allowed = same_block | causal
    mask = torch.zeros((1, 1, length, length), dtype=dtype, device=values.device)
    return mask.masked_fill_(~allowed[None, None], float("-inf"))

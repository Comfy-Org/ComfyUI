import torch

from comfy_extras.nodes_post_processing import MergeBatchListNode


def merge_batches(inputs, mode, overlap_frames=2):
    return MergeBatchListNode.execute(
        inputs,
        {"overlap": [mode], "overlap_frames": [overlap_frames]},
    )[0]


def test_merge_batch_list_trims_only_join_overlaps():
    first = torch.tensor([0, 1, 2, 3]).reshape(-1, 1, 1, 1)
    second = torch.tensor([10, 11, 12, 13]).reshape(-1, 1, 1, 1)

    assert merge_batches([first, second], "start").flatten().tolist() == [
        0,
        1,
        2,
        3,
        12,
        13,
    ]
    assert merge_batches([first, second], "end").flatten().tolist() == [
        0,
        1,
        10,
        11,
        12,
        13,
    ]
    assert merge_batches([first, second], "fade_linear").flatten().tolist() == [
        0,
        1,
        2,
        11,
        12,
        13,
    ]


def test_merge_batch_list_smooth_fade_and_single_batch():
    first = torch.zeros(6, 1, 1, 1)
    second = torch.full((6, 1, 1, 1), 10.0)
    expected = torch.tensor([0, 0, 0, 70 / 27, 200 / 27, 10, 10, 10])

    torch.testing.assert_close(
        merge_batches([first, second], "fade_smooth", overlap_frames=4).flatten(),
        expected,
    )

    batch = torch.arange(4).reshape(-1, 1, 1, 1)
    for mode in ("start", "end", "fade_linear", "fade_smooth"):
        torch.testing.assert_close(merge_batches([batch], mode), batch)

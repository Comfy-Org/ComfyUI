import gc
import os
import weakref
from fractions import Fraction

import torch

from comfy_api.input_impl.video_types import VideoFromComponents, VideoFromFile, VideoFromList
from comfy_api.input.basic_types import AudioInput
from comfy_api.util.video_types import VideoCodec, VideoComponents
from comfy_extras.nodes_video import ConcatenateVideo


def test_tensor_video_encodes_to_list_owned_file():
    images = torch.zeros((2, 16, 16, 3))
    images_ref = weakref.ref(images)
    source = VideoFromComponents(VideoComponents(images=images, frame_rate=Fraction(8)))

    video = VideoFromList([source])
    encoded = video.videos[0]
    path = encoded.get_stream_source()
    trimmed = video.as_trimmed(0, 0.125)
    del images, source, video, encoded
    gc.collect()

    assert isinstance(trimmed, VideoFromList)
    assert images_ref() is None
    assert os.path.exists(path)


def test_accumulate_flattens_groups_and_eagerly_encodes_tensors():
    images = [torch.full((1, 16, 16, 3), value) for value in (0.1, 0.5, 0.9)]
    references = [weakref.ref(image) for image in images]
    videos = [VideoFromComponents(VideoComponents(images=image, frame_rate=Fraction(8))) for image in images]
    nested = VideoFromList(videos[:2])

    result = ConcatenateVideo.execute({"inputs0": [nested], "inputs1": [videos[2]]}).result[0]
    del images, videos, nested
    gc.collect()

    assert len(result.videos) == 3
    assert all(isinstance(video, VideoFromFile) for video in result.videos)
    assert all(reference() is None for reference in references)


def test_concatenate_video_schema_and_intermediate_codec(monkeypatch):
    encoded_codecs = []

    def record_save(self, path, **kwargs):
        encoded_codecs.append(kwargs["codec"])
        with open(path, "wb"):
            pass

    monkeypatch.setattr(VideoFromComponents, "save_to", record_save)
    source = VideoFromComponents(
        VideoComponents(images=torch.zeros((1, 16, 16, 3)), frame_rate=Fraction(8))
    )
    ConcatenateVideo.execute({"inputs0": [source]}, codec=["av1"])

    schema = ConcatenateVideo.define_schema()
    inputs = {input.id: input for input in schema.inputs}
    assert encoded_codecs == [VideoCodec.AV1]
    assert inputs["codec"].advanced and inputs["complete_audio"].advanced
    assert schema.description and schema.outputs[0].tooltip
    assert all(input.tooltip for input in schema.inputs)
    assert inputs["inputs"].template.input.tooltip


def test_nested_complete_audio_uses_most_recent_override():
    source = VideoFromComponents(
        VideoComponents(images=torch.zeros((1, 16, 16, 3)), frame_rate=Fraction(8))
    )
    audios = [
        {"waveform": torch.full((1, 1, 1000), value), "sample_rate": 8000}
        for value in (1, 2, 3)
    ]
    nested = [VideoFromList([source], audio) for audio in audios[:2]]

    assert VideoFromList(nested).complete_audio is audios[1]
    assert VideoFromList(nested, audios[2]).complete_audio is audios[2]

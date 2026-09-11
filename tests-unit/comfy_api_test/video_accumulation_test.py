import io
import gc
import os
import tempfile
import weakref
from fractions import Fraction

import av
import torch

from comfy_api.input_impl.video_types import VideoFromComponents, VideoFromFile, VideoFromList
from comfy_api.input.basic_types import AudioInput
from comfy_api.util.video_types import VideoCodec, VideoComponents
from comfy_extras.nodes_video import ConcatenateVideo, CreateVideo


def test_tensor_video_encodes_to_list_owned_buffer():
    images = torch.zeros((2, 16, 16, 3))
    images_ref = weakref.ref(images)
    source = VideoFromComponents(VideoComponents(images=images, frame_rate=Fraction(8)))

    video = VideoFromList([source])
    encoded = video.videos[0]
    buffer = encoded.get_stream_source()
    trimmed = video.as_trimmed(0, 0.125)
    del images, source, video, encoded
    gc.collect()

    assert isinstance(trimmed, VideoFromList)
    assert images_ref() is None
    assert isinstance(buffer, io.BytesIO)
    assert buffer.getbuffer().nbytes > 0


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
        path.write(b"")

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


def test_create_video_optional_eager_encoding(monkeypatch):
    encoded_codecs = []

    def record_save(self, path, **kwargs):
        encoded_codecs.append(kwargs["codec"])
        path.write(b"")

    monkeypatch.setattr(VideoFromComponents, "save_to", record_save)
    video = CreateVideo.execute(torch.zeros((1, 16, 16, 3)), 8, codec="av1").result[0]

    codec_input = next(input for input in CreateVideo.define_schema().inputs if input.id == "codec")
    assert isinstance(video, VideoFromList)
    assert encoded_codecs == [VideoCodec.AV1]
    assert codec_input.options == ["none", "auto", "h264", "av1"]
    assert codec_input.default == "none"
    assert codec_input.advanced and codec_input.optional


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


def test_accumulated_video_packet_concatenates_file_backed_inputs():
    class NoMaterializeVideo(VideoFromFile):
        def get_components(self):
            raise AssertionError("file-backed concatenation decoded video frames")

        def save_to(self, *args, **kwargs):
            raise AssertionError("compatible file-backed video was rewritten")

    with tempfile.TemporaryDirectory() as directory:
        sources = []
        for index, extension in enumerate(("mkv", "mp4")):
            source = os.path.join(directory, f"source{index}.{extension}")
            VideoFromComponents(
                VideoComponents(images=torch.full((2, 16, 16, 3), index / 2), frame_rate=Fraction(8))
            ).save_to(source)
            sources.append(NoMaterializeVideo(source))

        output = os.path.join(directory, "output.mp4")
        VideoFromList(sources).save_to(output)
        with av.open(output) as container:
            assert sum(1 for _ in container.decode(video=0)) == 4


def test_accumulated_video_reencodes_all_chunks_with_shared_configuration():
    class RewriteTrackingVideo(VideoFromFile):
        rewritten = False

        def _save_transcoded(self, *args, **kwargs):
            self.rewritten = True
            return super()._save_transcoded(*args, **kwargs)

    with tempfile.TemporaryDirectory() as directory:
        sources = []
        for bit_depth in (8, 10):
            source = os.path.join(directory, f"source-{bit_depth}-bit.mp4")
            VideoFromComponents(
                VideoComponents(images=torch.zeros((2, 16, 16, 3)), frame_rate=Fraction(8)),
                bit_depth=bit_depth,
            ).save_to(source)
            sources.append(RewriteTrackingVideo(source))

        output = os.path.join(directory, "output.mp4")
        VideoFromList(sources).save_to(output)

        assert all(source.rewritten for source in sources)
        with av.open(output) as container:
            assert sum(1 for _ in container.decode(video=0)) == 4


def test_accumulated_video_reencodes_audio_to_shared_rate_and_layout():
    with tempfile.TemporaryDirectory() as directory:
        sources = []
        for index, (sample_rate, channels) in enumerate(((8000, 1), (16000, 2))):
            source = os.path.join(directory, f"source-{index}.mp4")
            audio = AudioInput({
                "waveform": torch.zeros((1, channels, sample_rate // 4)),
                "sample_rate": sample_rate,
            })
            VideoFromComponents(
                VideoComponents(
                    images=torch.zeros((2, 16, 16, 3)),
                    frame_rate=Fraction(8),
                    audio=audio,
                )
            ).save_to(source)
            sources.append(VideoFromFile(source))

        output = os.path.join(directory, "output.mp4")
        VideoFromList(sources).save_to(output)

        with av.open(output) as container:
            assert container.streams.audio[0].sample_rate == 8000
            assert container.streams.audio[0].layout.name == "mono"


def test_accumulated_video_stream_source_is_buffered_and_reused():
    video = VideoFromList([
        VideoFromComponents(VideoComponents(images=torch.zeros((1, 16, 16, 3)), frame_rate=Fraction(8)))
    ])

    first = video.get_stream_source()
    second = video.get_stream_source()

    assert first == second
    assert isinstance(first, io.BytesIO)
    assert first.getbuffer().nbytes > 0


def test_accumulated_video_continuously_encodes_audio_and_allows_override():
    audio = AudioInput({"waveform": torch.zeros((1, 2, 2000)), "sample_rate": 8000})
    videos = [
        VideoFromComponents(
            VideoComponents(images=torch.zeros((2, 16, 16, 3)), frame_rate=Fraction(8), audio=audio)
        )
        for _ in range(2)
    ]
    override = AudioInput({"waveform": torch.ones((1, 1, 8000)), "sample_rate": 8000})

    with tempfile.TemporaryDirectory() as directory:
        embedded_path = os.path.join(directory, "embedded.mp4")
        override_path = os.path.join(directory, "override.mp4")
        VideoFromList(videos).save_to(embedded_path)
        VideoFromList(videos, override).save_to(override_path)

        with av.open(embedded_path) as embedded, av.open(override_path) as overridden:
            embedded_audio = embedded.streams.audio[0]
            overridden_audio = overridden.streams.audio[0]
            assert embedded_audio.layout.name == "stereo"
            assert overridden_audio.layout.name == "mono"
            assert float(embedded_audio.duration * embedded_audio.time_base) <= 0.6
            assert float(overridden_audio.duration * overridden_audio.time_base) <= 0.6


def test_accumulated_video_metadata_and_explicit_materialization():
    videos = [
        VideoFromComponents(
            VideoComponents(images=torch.full((2, 16, 16, 3), value), frame_rate=Fraction(8))
        )
        for value in (0.0, 0.5)
    ]
    video = VideoFromList(videos)

    assert video.get_dimensions() == (16, 16)
    assert video.get_duration() == 0.5
    assert video.get_frame_count() == 4
    assert video.get_frame_rate() == 8
    assert video.get_components().images.shape == (4, 16, 16, 3)


def test_accumulated_video_reports_each_incompatible_dimension():
    videos = [
        VideoFromComponents(
            VideoComponents(images=torch.zeros((1, height, width, 3)), frame_rate=Fraction(8))
        )
        for width, height in ((16, 16), (24, 16), (16, 24))
    ]
    video = VideoFromList(videos)

    try:
        video.get_dimensions()
    except ValueError as error:
        assert str(error) == (
            "Accumulated videos have incompatible frame dimensions: "
            "chunk 0 is 16x16; chunk 1 is 24x16; chunk 2 is 16x24"
        )
    else:
        raise AssertionError("Expected incompatible dimensions to fail")


def test_accumulated_video_trims_across_file_boundaries_without_materializing():
    class NoMaterializeVideo(VideoFromFile):
        def get_components(self):
            raise AssertionError("trim materialized video frames")

    with tempfile.TemporaryDirectory() as directory:
        sources = []
        for index in range(2):
            source = os.path.join(directory, f"source{index}.mp4")
            VideoFromComponents(
                VideoComponents(images=torch.zeros((2, 16, 16, 3)), frame_rate=Fraction(8))
            ).save_to(source)
            sources.append(NoMaterializeVideo(source))

        trimmed = VideoFromList(sources).as_trimmed(0.125, 0.25, strict_duration=True)

        assert isinstance(trimmed, VideoFromList)
        assert len(trimmed.videos) == 2
        assert trimmed.get_duration() == 0.25


def test_accumulated_video_trim_slices_complete_audio():
    video = VideoFromList(
        [
            VideoFromComponents(
                VideoComponents(images=torch.zeros((4, 16, 16, 3)), frame_rate=Fraction(4))
            )
        ],
        AudioInput({"waveform": torch.arange(8000).reshape(1, 1, -1), "sample_rate": 8000}),
    )

    trimmed = video.as_trimmed(0.25, 0.5)

    assert torch.equal(trimmed.complete_audio["waveform"], torch.arange(2000, 6000).reshape(1, 1, -1))

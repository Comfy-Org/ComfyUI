"""ProRes proxies: H.264 in a faststart MP4 that every browser plays, tonemapped to SDR."""

import av
import pytest
from av.video.reformatter import ColorTrc

from app.assets.services import video_proxy

from .preview_helpers import write_prores


def _first_pixel(path) -> list[int]:
    with av.open(str(path)) as container:
        return next(container.decode(video=0)).to_ndarray(format="rgb24")[0, 0].tolist()


@pytest.mark.parametrize(("size", "proxy"), [((3840, 2160), (1920, 1080)), ((1080, 1920), (1080, 1920)), ((2160, 3840), (1080, 1920)), ((641, 361), (640, 360))])
def test_proxies_fit_1080p_with_even_sides_and_never_upscale(size, proxy):
    assert video_proxy._proxy_size(*size) == proxy


def test_a_proxy_is_browser_playable_h264(roots, tmp_path):
    source = write_prores(tmp_path / "clip.mov", 64, 48, frames=3, alpha=True, audio=True)
    ref = video_proxy.write_video_proxy(str(source))

    path = roots / "previews" / ref["filename"]
    assert ref == {"filename": ref["filename"], "width": 64, "height": 48}
    assert [p.name for p in (roots / "previews").iterdir()] == [ref["filename"]], "no temp file is left"
    data = path.read_bytes()
    assert data.find(b"moov") < data.find(b"mdat"), "faststart"
    with av.open(str(path)) as container:
        video, audio = container.streams.video[0], container.streams.audio[0]
        assert (video.codec_context.name, video.codec_context.pix_fmt, video.frames) == ("h264", "yuv420p", 3)
        assert (video.codec_context.color_primaries, video.codec_context.color_trc) == (1, ColorTrc.IEC61966_2_1)
        assert audio.codec_context.name == "aac"


def test_hdr_is_tonemapped_not_clipped(tmp_path):
    for name, transfer in (("pq", ColorTrc.SMPTE2084), ("hlg", ColorTrc.ARIB_STD_B67)):
        dest = tmp_path / f"{name}.mp4"
        video_proxy._transcode(str(write_prores(tmp_path / f"{name}.mov", 16, 16, value=0.75, transfer=transfer)), str(dest))
        assert all(100 < c < 240 for c in _first_pixel(dest)), name


def test_sdr_keeps_its_values(tmp_path):
    dest = tmp_path / "sdr.mp4"
    video_proxy._transcode(str(write_prores(tmp_path / "sdr.mov", 16, 16, value=0.5)), str(dest))
    assert all(abs(c - 128) <= 2 for c in _first_pixel(dest))

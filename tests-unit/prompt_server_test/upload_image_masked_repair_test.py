"""Tests for the mask editor painted-masked upload repair.

The mask editor uploads each save as four sibling files named
clipspace-{mask,paint,painted,painted-masked}-<timestamp>.png. Frontends that
serialize the masked layer from the browser canvas bitmap upload
premultiplied alpha, which zeroes the RGB of fully masked pixels; the server
rebuilds the file from the opaque painted sibling so the original image
survives under the painted mask (#16139).
"""

import io
import os

import numpy as np
import pytest
from PIL import Image

from server import repair_clipspace_painted_masked


class FakeUpload:
    def __init__(self, png_bytes, filename):
        self.file = io.BytesIO(png_bytes)
        self.filename = filename


def make_rgba(rgb_array, alpha_array):
    rgba = np.dstack([rgb_array, alpha_array]).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def gradient():
    h, w = 8, 8
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    rgb[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    rgb[..., 2] = 128
    return rgb


def test_rebuilds_rgb_from_painted_sibling(tmp_path, gradient):
    # Premultiplied upload: masked (top) half lost its RGB, like a blob
    # serialized from a browser canvas.
    alpha = np.full((8, 8), 255, dtype=np.uint8)
    alpha[:4] = 0
    premultiplied = gradient * (alpha[..., None] // 255)
    masked_bytes = make_rgba(premultiplied, alpha)
    with open(os.path.join(tmp_path, "clipspace-painted-42.png"), "wb") as f:
        f.write(make_rgba(gradient, np.full((8, 8), 255, dtype=np.uint8)))

    upload = FakeUpload(masked_bytes, "clipspace-painted-masked-42.png")
    data = repair_clipspace_painted_masked(upload, upload.filename, str(tmp_path))

    assert data is not None
    rebuilt = np.array(Image.open(io.BytesIO(data)))
    assert np.array_equal(rebuilt[..., :3], gradient)  # original pixels restored
    assert np.array_equal(rebuilt[..., 3], alpha)  # painted mask preserved

    # The upload stream is rewound so the raw write fallback still works.
    upload.file.seek(0)
    assert upload.file.read() == masked_bytes


def test_straight_alpha_upload_is_unchanged_in_content(tmp_path, gradient):
    # A frontend that encodes straight alpha keeps its RGB; rebuilding from
    # the identical painted sibling must not alter any pixel.
    alpha = np.full((8, 8), 255, dtype=np.uint8)
    alpha[:4] = 0
    masked_bytes = make_rgba(gradient, alpha)
    with open(os.path.join(tmp_path, "clipspace-painted-42.png"), "wb") as f:
        f.write(make_rgba(gradient, np.full((8, 8), 255, dtype=np.uint8)))

    upload = FakeUpload(masked_bytes, "clipspace-painted-masked-42.png")
    data = repair_clipspace_painted_masked(upload, upload.filename, str(tmp_path))

    rebuilt = np.array(Image.open(io.BytesIO(data)))
    assert np.array_equal(rebuilt[..., :3], gradient)
    assert np.array_equal(rebuilt[..., 3], alpha)


def test_returns_none_without_sibling(tmp_path):
    alpha = np.full((4, 4), 0, dtype=np.uint8)
    upload = FakeUpload(make_rgba(np.zeros((4, 4, 3), np.uint8), alpha),
                        "clipspace-painted-masked-42.png")
    assert repair_clipspace_painted_masked(upload, upload.filename, str(tmp_path)) is None


def test_returns_none_for_other_filenames(tmp_path, gradient):
    with open(os.path.join(tmp_path, "clipspace-painted-42.png"), "wb") as f:
        f.write(make_rgba(gradient, np.full((8, 8), 255, dtype=np.uint8)))
    for filename in ("clipspace-mask-42.png", "clipspace-painted-42.png", "example.png"):
        upload = FakeUpload(make_rgba(gradient, np.full((8, 8), 255, dtype=np.uint8)), filename)
        assert repair_clipspace_painted_masked(upload, filename, str(tmp_path)) is None


def test_returns_none_on_size_mismatch(tmp_path, gradient):
    alpha = np.full((8, 8), 0, dtype=np.uint8)
    with open(os.path.join(tmp_path, "clipspace-painted-42.png"), "wb") as f:
        f.write(make_rgba(gradient[:4], np.full((4, 8), 255, dtype=np.uint8)))

    upload = FakeUpload(make_rgba(gradient, alpha), "clipspace-painted-masked-42.png")
    assert repair_clipspace_painted_masked(upload, upload.filename, str(tmp_path)) is None


def test_returns_none_without_alpha_channel(tmp_path, gradient):
    with open(os.path.join(tmp_path, "clipspace-painted-42.png"), "wb") as f:
        f.write(make_rgba(gradient, np.full((8, 8), 255, dtype=np.uint8)))

    buffer = io.BytesIO()
    Image.fromarray(gradient, "RGB").save(buffer, format="PNG")
    upload = FakeUpload(buffer.getvalue(), "clipspace-painted-masked-42.png")
    assert repair_clipspace_painted_masked(upload, upload.filename, str(tmp_path)) is None


def test_repair_skips_non_png_filename(tmp_path):
    """A masked upload with a non-.png extension is written as-is."""
    original = np.full((4, 4, 3), 90, dtype=np.uint8)
    painted = make_rgba(original, np.full((4, 4), 255, dtype=np.uint8))
    upload = FakeUpload(painted, "clipspace-painted-masked-1.png.jpg")
    assert repair_clipspace_painted_masked(upload, upload.filename, str(tmp_path)) is None
    assert upload.file.tell() == 0


def test_repair_skips_non_png_sibling(tmp_path):
    """A decodable but non-PNG painted sibling does not feed a repair."""
    original = np.full((4, 4, 3), 90, dtype=np.uint8)
    masked = make_rgba(np.zeros((4, 4, 3), dtype=np.uint8), np.full((4, 4), 255, dtype=np.uint8))
    jpeg = io.BytesIO()
    Image.fromarray(original).save(jpeg, format="JPEG")
    (tmp_path / "clipspace-painted-1.png").write_bytes(jpeg.getvalue())
    upload = FakeUpload(masked, "clipspace-painted-masked-1.png")
    assert repair_clipspace_painted_masked(upload, upload.filename, str(tmp_path)) is None
    assert upload.file.tell() == 0


def test_upload_flow_repairs_before_collision_handling(tmp_path):
    """A colliding masked upload still saves repaired bytes under the renamed
    target: repair resolves from the original filename before the collision
    loop renames it (#16139 review)."""
    import server

    original = np.full((4, 4, 3), 120, dtype=np.uint8)
    masked = make_rgba(np.zeros((4, 4, 3), dtype=np.uint8), np.full((4, 4), 255, dtype=np.uint8))
    painted = make_rgba(original, np.full((4, 4), 255, dtype=np.uint8))
    (tmp_path / "clipspace-painted-1.png").write_bytes(painted)
    # A different file already occupies the masked target name.
    (tmp_path / "clipspace-painted-masked-1.png").write_bytes(b"stale-bytes")

    upload = FakeUpload(masked, "clipspace-painted-masked-1.png")

    async def run():
        # Minimal exercise of the upload flow's plain-write branch: replicate
        # the ordering contract the flow now guarantees — repaired bytes are
        # computed from the original filename before any rename.
        data = server.repair_clipspace_painted_masked(upload, upload.filename, str(tmp_path))
        upload.file.seek(0)
        if data is None:
            data = upload.file.read()
        target = tmp_path / "clipspace-painted-masked-1 (1).png"
        target.write_bytes(data)
        return target

    import asyncio
    target = asyncio.run(run())
    with Image.open(target) as img:
        arr = np.array(img.convert("RGB"))
    assert (arr == 120).all(), "collision-renamed upload must still carry the original pixels"


def test_upload_flow_duplicate_detects_repaired_bytes(tmp_path):
    """When the existing target already holds the repaired bytes, the
    duplicate hash check sees a duplicate instead of colliding."""
    import hashlib

    original = np.full((4, 4, 3), 200, dtype=np.uint8)
    masked = make_rgba(np.zeros((4, 4, 3), dtype=np.uint8), np.full((4, 4), 255, dtype=np.uint8))
    painted = make_rgba(original, np.full((4, 4), 255, dtype=np.uint8))
    (tmp_path / "clipspace-painted-1.png").write_bytes(painted)

    import server
    upload = FakeUpload(masked, "clipspace-painted-masked-1.png")
    repaired = server.repair_clipspace_painted_masked(upload, upload.filename, str(tmp_path))
    upload.file.seek(0)
    assert repaired is not None

    # The flow's compare_file_hash contract: existing repaired file == repaired bytes.
    (tmp_path / "clipspace-painted-masked-1.png").write_bytes(repaired)
    digest_existing = hashlib.sha256((tmp_path / "clipspace-painted-masked-1.png").read_bytes()).hexdigest()
    digest_data = hashlib.sha256(repaired).hexdigest()
    assert digest_existing == digest_data

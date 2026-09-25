from app.assets.services.file_utils import get_size_and_mtime_ns


def test_file_stat_reports_size_and_mtime(tmp_path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"bytes")

    size, mtime_ns = get_size_and_mtime_ns(str(path))

    assert size == 5
    assert mtime_ns > 0

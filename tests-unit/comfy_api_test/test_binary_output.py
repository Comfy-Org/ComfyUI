import asyncio
from pathlib import Path

import pytest

import folder_paths
from comfy_api.latest._sdk import _InProcessOutput


@pytest.fixture
def output(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, 'get_output_directory', lambda: str(tmp_path))
    return _InProcessOutput(), tmp_path


async def stream(*chunks):
    for chunk in chunks:
        yield chunk


def test_complete_bytes_and_relative_name(output):
    provider, root = output
    chunks = [bytes(range(256)) * 4096, b'GGUF\x03\x00\x00\x00', b'end']
    name = asyncio.run(provider.save_binary(stream(*chunks), 'weights/test',
                                           max_bytes=sum(map(len, chunks))))
    assert not Path(name).is_absolute()
    assert (root / name).read_bytes() == b''.join(chunks)
    assert sorted(p.name for p in (root / 'weights').iterdir()) == [Path(name).name]


def test_existing_file_is_preserved(output):
    provider, root = output
    (root / 'weights_00001_.gguf').write_bytes(b'original')
    name = asyncio.run(provider.save_binary(stream(b'new'), 'weights', max_bytes=3))
    assert (root / 'weights_00001_.gguf').read_bytes() == b'original'
    assert (root / name).read_bytes() == b'new'
    assert name != 'weights_00001_.gguf'


@pytest.mark.parametrize('chunk', [b'', 'not bytes', bytearray(b'a'), b'x' * (1024**2 + 1)],
                         ids=['empty', 'text', 'mutable', 'oversize'])
def test_invalid_chunk_leaves_no_partial(output, chunk):
    provider, root = output
    with pytest.raises(ValueError, match='chunks'):
        asyncio.run(provider.save_binary(stream(b'first', chunk), 'test', max_bytes=2**22))
    assert list(root.iterdir()) == []


def test_limit_is_checked_before_excess_bytes_are_written(output):
    provider, root = output
    with pytest.raises(ValueError, match='max_bytes'):
        asyncio.run(provider.save_binary(stream(b'abc', b'def'), 'test', max_bytes=5))
    assert list(root.iterdir()) == []


@pytest.mark.parametrize('budget', [True, 0, -1, 1.0, 2**63])
def test_invalid_budget_does_not_consume_source(output, budget):
    provider, root = output
    async def unexpected():
        raise AssertionError('invalid request consumed source')
        yield b'x'
    with pytest.raises(ValueError, match='max_bytes'):
        asyncio.run(provider.save_binary(unexpected(), 'test', max_bytes=budget))
    assert list(root.iterdir()) == []


@pytest.mark.parametrize('prefix,extension', [('../escape', '.gguf'), ('/escape', '.bin'),
                                            ('valid', '.py'), ('valid', '../file.gguf')])
def test_filename_boundary(output, prefix, extension):
    provider, root = output
    with pytest.raises(ValueError):
        asyncio.run(provider.save_binary(stream(b'x'), prefix, max_bytes=1, extension=extension))
    assert list(root.iterdir()) == []


def test_symlink_directory_is_refused(output, tmp_path):
    provider, root = output
    outside = tmp_path.parent / (tmp_path.name + '-outside')
    outside.mkdir()
    (root / 'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        asyncio.run(provider.save_binary(stream(b'x'), 'escape/file', max_bytes=1))
    assert list(outside.iterdir()) == []


def test_source_exception_removes_partial(output):
    provider, root = output
    async def broken():
        yield b'first'
        raise LookupError('source broke')
    with pytest.raises(LookupError, match='source broke'):
        asyncio.run(provider.save_binary(broken(), 'test', max_bytes=32))
    assert list(root.iterdir()) == []


def test_cancellation_removes_partial(output):
    provider, root = output
    async def run():
        ready = asyncio.Event()
        async def waiting():
            yield b'first'
            ready.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(provider.save_binary(waiting(), 'test', max_bytes=32))
        await ready.wait()
        assert any(p.name.startswith('.comfy-binary-') for p in root.iterdir())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
    assert list(root.iterdir()) == []

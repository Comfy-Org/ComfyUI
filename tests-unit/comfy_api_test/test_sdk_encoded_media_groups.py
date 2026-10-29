import asyncio
import os

import pytest

from comfy_api.latest import io
from comfy_api.latest._sdk import (
    ExecutionPlan, InProcessCtxProvider, InProcessOps, InProcessRefResolver,
    MediaGroupRef, Ref, ValueRef, bind_runtime, unwrap_outputs,
)


PNG = b"\x89PNG\r\n\x1a\nencoded fixture bytes"
pytestmark = pytest.mark.skipif(
    not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY"))
    or not {os.open, os.mkdir, os.stat, os.link, os.unlink, os.rmdir}.issubset(os.supports_dir_fd),
    reason="managed media groups require descriptor-relative filesystem support",
)


def runtime(tmp_path, monkeypatch):
    import folder_paths

    output, temporary = tmp_path / "output", tmp_path / "temp"
    output.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(temporary))
    refs = InProcessRefResolver()
    context = InProcessCtxProvider().build(ExecutionPlan(
        prompt_id="encoded-media", node_id="1", node_type="encoded-media",
        prompt={"1": {"class_type": "encoded-media"}},
    ))
    return refs, context, InProcessOps()


async def chunks(data):
    yield data


async def group(context, **options):
    return await context.output.reserve_media_group(
        "video/example", max_bytes=4096, max_entries=4, timeout_seconds=2,
        failed_entry_policy="retain", **options,
    )


@pytest.mark.parametrize("save_output", [True, False])
def test_named_absent_entry_and_exact_incremental_publication(tmp_path, monkeypatch, save_output):
    async def run():
        refs, context, ops = runtime(tmp_path, monkeypatch)
        with bind_runtime(refs, context, ops):
            async with await group(context, save_output=save_output) as lease:
                declared = await lease.reserve("utility", ".png", "utility")
                root = tmp_path / ("output" if save_output else "temp")
                assert not declared["materialized"]
                assert not (root / declared["name"]).exists()
                published = await lease.publish("utility", chunks(PNG), max_bytes=4096)
                assert published["bytes_written"] == len(PNG)
                assert published["state"] == "committed"
                assert (root / declared["name"]).read_bytes() == PNG
                final = await lease.reserve("final", ".unknown", "final")
                with pytest.raises(ValueError, match="extension"):
                    await lease.publish("final", chunks(b"bad"), max_bytes=4000)
                assert (root / declared["name"]).read_bytes() == PNG
                assert not (root / final["name"]).exists()
        refs.clear()

    asyncio.run(run())


@pytest.mark.parametrize("policy,retained", [("retain", True), ("discard", False)])
def test_failure_policy_is_explicit_and_preserves_original_error(tmp_path, monkeypatch, policy, retained):
    async def failing():
        yield PNG
        raise RuntimeError("encoder failure after bytes")

    async def run():
        refs, context, ops = runtime(tmp_path, monkeypatch)
        with bind_runtime(refs, context, ops):
            lease = await context.output.reserve_media_group(
                "video/example", max_bytes=4096, max_entries=4,
                timeout_seconds=2, failed_entry_policy=policy,
            )
            declared = await lease.reserve("final", ".png")
            with pytest.raises(RuntimeError, match="encoder failure after bytes"):
                await lease.publish("final", failing(), max_bytes=4096)
            record = (await lease.entries())[0]
            assert record["state"] == "failed"
            assert record["materialized"] is retained
            assert record["bytes_written"] == len(PNG)
            target = tmp_path / "output" / declared["name"]
            assert target.exists() is retained
            if retained:
                assert target.read_bytes() == PNG
            await lease.close()
        refs.clear()

    asyncio.run(run())


def test_replay_does_not_consume_again_and_removal_keeps_listed_name(tmp_path, monkeypatch):
    async def forbidden():
        raise AssertionError("committed replay consumed a stream")
        yield b""

    async def run():
        refs, context, ops = runtime(tmp_path, monkeypatch)
        with bind_runtime(refs, context, ops):
            async with await group(context) as lease:
                declaration = await lease.reserve("base", ".png", "intermediate")
                first = await lease.publish("base", chunks(PNG), max_bytes=4096)
                assert await lease.publish("base", forbidden(), max_bytes=4096) == first
                removed = await lease.remove("base")
                assert removed["name"] == declaration["name"]
                assert removed["state"] == "deleted"
                assert not (tmp_path / "output" / declaration["name"]).exists()
                assert (await lease.entries())[0]["name"] == declaration["name"]
        refs.clear()

    asyncio.run(run())


def test_unique_group_counter_and_managed_writers_cannot_overwrite(tmp_path, monkeypatch):
    import folder_paths

    async def run():
        refs, context, ops = runtime(tmp_path, monkeypatch)
        with bind_runtime(refs, context, ops):
            one, two = await group(context), await group(context)
            first = await one.reserve("final", ".png")
            second = await two.reserve("final", ".png")
            assert first["name"] != second["name"]
            await one.publish("final", chunks(PNG), max_bytes=4096)
            with pytest.raises(PermissionError, match="owning lease"):
                await context.output.write_text("replace", first["name"], mode="overwrite")
            with pytest.raises(PermissionError, match="owning lease"):
                folder_paths.get_save_image_path(first["name"], str(tmp_path / "output"))
            alias = tmp_path / "output" / "alias"
            alias.symlink_to(tmp_path / "output" / ".comfy-media", target_is_directory=True)
            with pytest.raises(PermissionError, match="owning lease"):
                folder_paths.get_save_image_path("alias/example", str(tmp_path / "output"))
            assert (tmp_path / "output" / first["name"]).read_bytes() == PNG
            await one.close()
            await two.close()
        refs.clear()

    asyncio.run(run())


def test_scope_retirement_revokes_handles_and_cancels_pending_stream(tmp_path, monkeypatch):
    entered = asyncio.Event()
    suspended = asyncio.Event()

    async def pending():
        yield PNG
        entered.set()
        await suspended.wait()

    async def run():
        refs, context, ops = runtime(tmp_path, monkeypatch)
        with bind_runtime(refs, context, ops):
            lease = await group(context)
            await lease.reserve("final", ".png")
            owned = await refs.resolve(lease)
            descriptor = owned._directory
            task = asyncio.create_task(lease.publish("final", pending(), max_bytes=4096))
            await entered.wait()
            refs.clear()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(OSError):
                os.fstat(descriptor)
            with pytest.raises(KeyError):
                await lease.entries()
        new, context, ops = runtime(tmp_path, monkeypatch)
        with bind_runtime(new, context, ops):
            with pytest.raises(KeyError):
                await MediaGroupRef._wrap(Ref("MEDIA_OUTPUT_GROUP", lease.id)).entries()
        new.clear()

    asyncio.run(run())


def test_group_cannot_escape_as_node_output_or_be_forged(tmp_path, monkeypatch):
    async def run():
        refs, context, ops = runtime(tmp_path, monkeypatch)
        with bind_runtime(refs, context, ops):
            lease = await group(context)
            with pytest.raises(TypeError, match="cannot escape"):
                await unwrap_outputs(refs, io.NodeOutput({"nested": [lease]}))
            wrapped = await ValueRef.from_value({"nested": [lease]})
            with pytest.raises(TypeError, match="cannot escape"):
                await unwrap_outputs(refs, io.NodeOutput(wrapped))
            forged = MediaGroupRef._wrap(await refs.create("MEDIA_OUTPUT_GROUP", {"root": "/"}))
            with pytest.raises(TypeError, match="owned reservation"):
                await forged.entries()
            with pytest.raises(ValueError, match="path"):
                await lease.reserve("final", "../outside.png")
        refs.clear()

    asyncio.run(run())


@pytest.mark.parametrize("unknown_size", [False, True])
def test_short_write_failure_records_observed_bytes_and_charges_unknown_size(tmp_path, monkeypatch, unknown_size):
    from comfy_api.latest import _encoded_media

    write, stat = os.write, os.stat
    attempts = 0

    def fail_after_short_write(fd, data):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return write(fd, data[:5])
        raise OSError("disk full after short write")

    def failed_stat(path, *args, **kwargs):
        if unknown_size and str(path).startswith(".partial-"):
            raise OSError("temporary size unavailable")
        return stat(path, *args, **kwargs)

    async def run():
        refs, context, ops = runtime(tmp_path, monkeypatch)
        with bind_runtime(refs, context, ops):
            async with await group(context) as lease:
                declared = await lease.reserve("final", ".png")
                with monkeypatch.context() as faults:
                    faults.setattr(_encoded_media.os, "write", fail_after_short_write)
                    faults.setattr(_encoded_media.os, "stat", failed_stat)
                    with pytest.raises(OSError, match="disk full after short write"):
                        await lease.publish("final", chunks(PNG), max_bytes=4096)
                record = (await lease.entries())[0]
                assert record["bytes_written"] == 5
                assert record["materialized"] is not unknown_size
                if unknown_size:
                    assert record["cleanup_error"] == "OSError"
                    assert record["retention_error"] == "OSError"
                    await lease.reserve("next", "_next.png")
                    with pytest.raises(ValueError, match="remaining group byte budget"):
                        await lease.publish("next", chunks(PNG), max_bytes=1)
                else:
                    assert (tmp_path / "output" / declared["name"]).read_bytes() == PNG[:5]
        refs.clear()

    asyncio.run(run())


def test_revoked_stream_cleanup_failure_is_logged_without_replacing_cancellation(tmp_path, monkeypatch, caplog):
    async def run():
        entered = asyncio.Event()

        class FailedCleanup:
            def __aiter__(self):
                return self

            async def __anext__(self):
                entered.set()
                await asyncio.Event().wait()

            async def aclose(self):
                raise LookupError("producer cleanup failed")

        refs, context, ops = runtime(tmp_path, monkeypatch)
        with bind_runtime(refs, context, ops):
            lease = await group(context)
            await lease.reserve("final", ".png")
            owned = await refs.resolve(lease)
            descriptor = owned._directory
            task = asyncio.create_task(lease.publish("final", FailedCleanup(), max_bytes=4096))
            await entered.wait()
            refs.clear()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert "media entry final cleanup=LookupError" in caplog.text
            with pytest.raises(OSError):
                os.fstat(descriptor)
            with pytest.raises(KeyError):
                await lease.entries()

    asyncio.run(run())


def test_expired_group_keeps_snapshots_but_refuses_writes(tmp_path, monkeypatch):
    async def run():
        refs, context, ops = runtime(tmp_path, monkeypatch)
        with bind_runtime(refs, context, ops):
            async with await group(context) as lease:
                await lease.reserve("final", ".png")
                published = await lease.publish("final", chunks(PNG), max_bytes=4096)
                owned = await refs.resolve(lease)
                owned._deadline = 0
                assert await lease.entries() == [published]
                with pytest.raises(TimeoutError, match="expired"):
                    await lease.remove("final")
                with pytest.raises(TimeoutError, match="expired"):
                    await lease.reserve("next", "_next.png")
        refs.clear()

    asyncio.run(run())


def test_casefolded_names_and_generated_length_are_refused_before_publication(tmp_path, monkeypatch):
    async def run():
        refs, context, ops = runtime(tmp_path, monkeypatch)
        with bind_runtime(refs, context, ops):
            async with await group(context) as lease:
                await lease.reserve("one", ".png")
                with pytest.raises(ValueError, match="filename is already reserved"):
                    await lease.reserve("two", ".PNG")
            async with await context.output.reserve_media_group(
                "x" * 128, max_bytes=4096, max_entries=4,
                timeout_seconds=2, failed_entry_policy="retain",
            ) as lease:
                with pytest.raises(ValueError, match="generated media filename"):
                    await lease.reserve("long", "_" + "x" * 127)
        refs.clear()

    asyncio.run(run())

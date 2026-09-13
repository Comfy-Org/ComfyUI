from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session as SASession

from app.assets import scanner, seeder as seeder_module
from app.assets.database.models import AssetContent
from app.assets.database.queries.records import create_content, create_record
from app.assets.helpers import to_stored_hash
from app.assets.services import hash_mode_state
from app.assets.services.hash_mode_state import (
    clear_transition_queue,
    enqueue_transition_work,
    read_stored_mode,
    record_transition_intent,
    write_stored_mode,
)

_ATTEMPT_BUDGET = 5


class _AttemptBudgetExhausted(BaseException):
    # BaseException, not Exception: enrich_assets_batch's blanket except would swallow it.
    pass


@pytest.fixture(autouse=True)
def transition_queue():
    clear_transition_queue()
    yield
    clear_transition_queue()


def _denied(_candidate_path: str):
    raise PermissionError("denied")


def test_enrich_phase_settles_an_unreadable_transition_without_looping(
    session, db_engine, temp_dir: Path, monkeypatch
):
    path = temp_dir / "unreadable.safetensors"
    payload = b"bytes that can be stat'd but never hashed"
    path.write_bytes(payload)
    stat = path.stat()
    content = create_content(
        session, str(path), to_stored_hash("seed-digest"), stat.st_size, stat.st_mtime_ns
    )
    content_id = content.id
    record = create_record(session, content_id, path.name)
    record_id = record.id
    write_stored_mode(session, "off")
    monkeypatch.setattr(hash_mode_state._mode, "hashing_enabled", lambda: True)

    transition = record_transition_intent(session)
    enqueue_transition_work(session, transition)
    session.commit()

    @contextmanager
    def _create_session():
        with SASession(db_engine) as sess:
            yield sess

    attempts: list[str] = []
    real_enrich_asset = scanner.enrich_asset

    def counting_enrich_asset(*args, **kwargs):
        attempts.append(kwargs["record_id"])
        if len(attempts) > _ATTEMPT_BUDGET:
            raise _AttemptBudgetExhausted
        return real_enrich_asset(*args, **kwargs)

    seeder = seeder_module._AssetSeeder()
    seeder._compute_hashes = True
    seeder._run_gate.set()
    seeder._cancel_event.clear()

    monkeypatch.setattr("folder_paths.get_input_directory", lambda: str(temp_dir))
    monkeypatch.setattr(hash_mode_state, "snapshot_hash", _denied)
    monkeypatch.setattr(scanner, "snapshot_hash", _denied)
    monkeypatch.setattr(scanner, "enrich_asset", counting_enrich_asset)

    with patch("app.assets.seeder.create_session", _create_session), \
         patch("app.assets.scanner.create_session", _create_session):
        try:
            cancelled, _enriched = seeder._run_enrich_phase(("input",))
        except _AttemptBudgetExhausted:
            pytest.fail(
                f"the enrich phase re-selected the same record more than {_ATTEMPT_BUDGET} "
                "times: a terminally-cleared row stays hash-eligible, so counting its "
                "metadata as progress keeps it out of failed_ids and the pass never ends"
            )

    assert cancelled is False
    session.expire_all()
    assert read_stored_mode(session) == "on", (
        "one background scan must settle the transition on its own; waiting for a future "
        "prompt to queue another enrich pass leaves a quiet server wedged at 'off'"
    )
    settled = session.get(AssetContent, content_id)
    assert settled.is_missing is False, (
        "an unreadable file is not a deleted one; settling must not mark its row missing"
    )
    assert settled.hash is None
    assert attempts == [record_id], (
        "the record is attempted once, then excluded from the rest of the pass"
    )

import pytest

from app.assets.database.queries.records import (
    create_content,
    create_record,
    mark_content_missing,
)
from app.assets.services.asset_management import get_export_file, list_job_export_files

JOB_A = "11111111-1111-4111-8111-111111111111"
JOB_B = "22222222-2222-4222-8222-222222222222"


def seed_job_records(session):
    records = {}
    for key, path, job_id, tags in [
        ("saved", "/output/shots/a.exr", JOB_A, ["output"]),
        ("preview", "/temp/p.png", JOB_A, ["temp"]),
        ("both", "/output/both.png", JOB_A, ["temp", "output"]),
        ("other_job", "/output/b.png", JOB_B, ["output"]),
        ("input", "/input/ref.png", JOB_A, ["input"]),
        ("missing", "/output/gone.png", JOB_A, ["output"]),
    ]:
        content = create_content(session, path, hash=f"blake3:{key}")
        records[key] = create_record(session, content.id, path.rsplit("/", 1)[-1], job_id=job_id, tags=tags)
    mark_content_missing(session, records["missing"].content_id)
    session.commit()
    return records


@pytest.mark.parametrize(
    "include_previews, expected",
    [
        (False, ["both", "saved"]),
        (True, ["both", "preview", "saved"]),
    ],
)
def test_job_export_files_are_live_outputs_plus_requested_previews(
    session, mock_create_session, include_previews, expected
) -> None:
    records = seed_job_records(session)
    key_by_id = {record.id: key for key, record in records.items()}

    files = list_job_export_files([JOB_A], include_previews)

    assert sorted(key_by_id[f.id] for f in files) == expected


def test_job_export_files_carry_the_content_path_and_hash(session, mock_create_session) -> None:
    records = seed_job_records(session)

    [saved] = [f for f in list_job_export_files([JOB_A], False) if f.id == records["saved"].id]

    assert saved.name == "a.exr"
    assert saved.job_id == JOB_A
    assert saved.path.replace("\\", "/").endswith("/output/shots/a.exr")
    assert saved.hash == "blake3:saved"


def test_job_export_files_for_no_jobs_is_empty(session, mock_create_session) -> None:
    seed_job_records(session)

    assert list_job_export_files([], True) == []


@pytest.mark.parametrize("key", ["saved", "preview", "input"])
def test_export_file_resolves_a_record(session, mock_create_session, key) -> None:
    records = seed_job_records(session)

    file = get_export_file(records[key].id)

    assert file.id == records[key].id
    assert file.name == records[key].name


@pytest.mark.parametrize(
    "reference, error",
    [("unknown", ValueError), ("missing", FileNotFoundError)],
)
def test_export_file_rejects_unknown_or_missing_records(session, mock_create_session, reference, error) -> None:
    records = seed_job_records(session)
    reference_id = records[reference].id if reference in records else "33333333-3333-4333-8333-333333333333"

    with pytest.raises(error):
        get_export_file(reference_id)

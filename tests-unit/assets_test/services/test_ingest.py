from pathlib import Path

from sqlalchemy import select

import folder_paths
from app.assets.database.models import Asset, AssetContent
from app.assets.services.ingest import register_executed_output


def test_output_registration_creates_separate_record_and_content_rows(mock_create_session) -> None:
    path = Path(folder_paths.get_output_directory()) / "result-b-schema.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("result")

    registered = register_executed_output(str(path), job_id="job")

    with mock_create_session() as session:
        record = session.get(Asset, registered.id)
        content = session.scalar(select(AssetContent).where(AssetContent.id == registered.content_id))
    assert record is not None
    assert content is not None
    assert record.content_id == content.id
    assert record.job_id == "job"

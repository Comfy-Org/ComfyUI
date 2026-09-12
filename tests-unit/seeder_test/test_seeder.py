from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.assets.database.models import Base
from app.assets.database.queries import create_content, create_record, mark_content_missing


def test_seeder_models_missing_as_content_state():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        content = create_content(session, "/models/checkpoints/model.safetensors", hash=None)
        record = create_record(
            session,
            content.id,
            "model.safetensors",
            loader_path="checkpoints/model.safetensors",
            tags=["models", "model_type:checkpoints"],
        )

        mark_content_missing(session, content.id)

        assert content.is_missing is True
        assert record.content_id == content.id

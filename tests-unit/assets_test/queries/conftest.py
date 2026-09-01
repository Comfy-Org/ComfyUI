import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.assets.database.models import Base


@pytest.fixture(scope="session", autouse=True)
def assert_asset_metadata_tables():
    assert set(Base.metadata.tables) == {
        "assets",
        "asset_contents",
        "asset_meta",
        "asset_tags",
        "tags",
        "asset_system_state",
    }


@pytest.fixture
def session():
    """In-memory SQLite session for fast unit tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        yield sess


@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    """Override parent autouse fixture - query tests don't need server cleanup."""
    yield

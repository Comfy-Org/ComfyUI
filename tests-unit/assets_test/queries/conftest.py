import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.assets.database.models import Base


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

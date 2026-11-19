from typing import Any
from datetime import datetime
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

def to_dict(obj: Any, include_none: bool = False) -> dict[str, Any]:
    fields = obj.__table__.columns.keys()
    out: dict[str, Any] = {}
    for field in fields:
        val = getattr(obj, field)
        if val is None and not include_none:
            continue
        if isinstance(val, datetime):
            out[field] = val.isoformat()
        else:
            out[field] = val
    return out

# TODO: Define models here

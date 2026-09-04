"""Re-exports the record and tag query functions so callers import them from one
place instead of reaching into individual query modules. A module-level
``__getattr__`` resolves names that live in the tag module, keeping this import
surface flat as queries are split across more files.
"""

from importlib import import_module

from app.assets.database.queries.records import (
    create_content,
    create_content_reporting_insert,
    create_record,
    delete_record,
    fetch_record_tags,
    get_record_by_id,
    list_records_page,
    mark_content_missing,
    rename_record,
    unset_content_missing,
    update_record_access_time,
)

__all__ = [
    "create_content",
    "create_content_reporting_insert",
    "create_record",
    "delete_record",
    "fetch_record_tags",
    "get_record_by_id",
    "list_records_page",
    "mark_content_missing",
    "rename_record",
    "unset_content_missing",
    "update_record_access_time",
]


def __getattr__(name: str):
    for module_name in ("tags",):
        module = import_module(f"app.assets.database.queries.{module_name}")
        candidate = getattr(module, name, None)
        if candidate is not None:
            return candidate
    raise AttributeError(name)

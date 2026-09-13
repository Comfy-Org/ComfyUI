"""Reset steps that bring stale asset rows forward to the current semantics.

Alembic migrates the shape of the assets tables; these steps migrate the meaning
of what is in them. A step must be idempotent, is applied in order from the
version stamped in the database, and is stamped only once it finishes, so an
interrupted run resumes rather than half-applying. How broad or surgical a step
is depends on the drift it repairs; the registry is what stays.
"""

import logging
import time

from app.assets.database.queries.semantics import (
    get_semantics_version,
    set_semantics_version,
)
from app.assets.semantics.reproject_derived import reproject_derived_state
from app.assets.semantics.step import (
    InterruptCheck,
    SemanticsStep,
    SemanticsStepInterrupted,
    StepResult,
)
from app.database.db import can_create_session, create_session

__all__ = [
    "CURRENT_SEMANTICS_VERSION",
    "SEMANTICS_STEPS",
    "InterruptCheck",
    "SemanticsStep",
    "SemanticsStepInterrupted",
    "StepResult",
    "run_pending_semantics_steps",
]

SEMANTICS_STEPS: tuple[SemanticsStep, ...] = (
    SemanticsStep(
        version=1,
        description="reproject path-derived reference state",
        apply=reproject_derived_state,
    ),
)

CURRENT_SEMANTICS_VERSION = SEMANTICS_STEPS[-1].version


def run_pending_semantics_steps(interrupt_check: InterruptCheck | None = None) -> int:
    """Returns steps applied. A database already at the current version costs one indexed row read."""
    if not can_create_session():
        return 0

    try:
        with create_session() as session:
            stored_version = get_semantics_version(session)
    except Exception:
        logging.exception(
            "Could not read the asset semantics version; skipping semantics reset"
        )
        return 0

    pending = [step for step in SEMANTICS_STEPS if step.version > stored_version]
    if not pending:
        return 0

    logging.info(
        "Asset semantics at version %d, current is %d: applying %d step(s)",
        stored_version,
        CURRENT_SEMANTICS_VERSION,
        len(pending),
    )

    applied = 0
    for step in pending:
        started = time.perf_counter()
        try:
            summary = step.apply(interrupt_check)
        except SemanticsStepInterrupted:
            logging.info(
                "Asset semantics step %d (%s) interrupted; resuming on next start",
                step.version,
                step.description,
            )
            return applied
        except Exception:
            logging.exception(
                "Asset semantics step %d (%s) failed; database stays at version %d",
                step.version,
                step.description,
                stored_version + applied,
            )
            return applied

        if not summary.complete:
            logging.info(
                "Asset semantics step %d (%s) left work unfinished, staying at "
                "version %d so it runs again: %s",
                step.version,
                step.description,
                stored_version + applied,
                summary,
            )
            return applied

        try:
            with create_session() as session:
                set_semantics_version(session, step.version)
                session.commit()
        except Exception:
            logging.exception(
                "Asset semantics step %d (%s) finished but could not be stamped, "
                "so it runs again; its own writes are already committed",
                step.version,
                step.description,
            )
            return applied

        applied += 1
        logging.info(
            "Asset semantics step %d (%s) applied in %.3fs: %s",
            step.version,
            step.description,
            time.perf_counter() - started,
            summary,
        )

    return applied

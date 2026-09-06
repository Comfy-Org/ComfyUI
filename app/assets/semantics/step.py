
from dataclasses import dataclass
from typing import Callable

InterruptCheck = Callable[[], bool]


@dataclass
class StepResult:
    """What a step did. A falsy ``complete`` withholds the stamp, so the step runs again."""

    @property
    def complete(self) -> bool:
        return True


class SemanticsStepInterrupted(Exception):
    """A step stopped early on request.

    Its partial work stands -- steps are idempotent, so a resumed run repeats it
    harmlessly -- but the version is not stamped, so the step runs again.
    """


@dataclass(frozen=True)
class SemanticsStep:
    """``apply`` must be idempotent, and must raise SemanticsStepInterrupted rather than return early."""

    version: int
    description: str
    apply: Callable[[InterruptCheck | None], StepResult]

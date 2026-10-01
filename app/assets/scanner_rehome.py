"""The startup prune's re-home step.

A row outside every registered folder by text may still sit in one reached under
another spelling: a symlink, junction, ``subst`` drive, 8.3 name, ``\\\\?\\`` prefix,
or a change of letter case. Such a row keeps its id and records, with its path
rewritten to the folder's spelling today. Every row the code can't prove gets the
outcome it would have without this step: retired, or left live if the platform's own
case rules still place it in a registered folder.

``plan_prune`` only reads; ``apply_prune_plan`` does every write.
"""

import logging
import os
import threading
from collections import Counter
from typing import Callable, NamedTuple

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetContent, AssetTag
from app.assets.database.queries import is_live_path_conflict, mark_content_missing
from app.assets.services.path_utils import compute_loader_path, get_backend_system_tags_from_path

# The filesystem reads stop after this long without progress; undecided rows get the
# outcome they would have without this step.
# A stall timeout rather than a deadline, as a large re-spelled folder is legitimately slow.
STALL_SECONDS = 10.0
_BATCH = 500
_ROOT_TAGS = frozenset({"input", "output", "temp", "models"})

# What a row's records say about where it belongs: root and model_type tags, loader path.
Role = tuple[frozenset[str], str | None]


class PrunePlan(NamedTuple):
    moves: dict[str, str]  # content id -> path under today's spelling
    retire: list[str]
    spared: frozenset[str] = frozenset()  # never retired, even if their move fails


class PruneResult(NamedTuple):
    marked: int
    rehomed: int


def _strip_verbatim(path: str) -> str:
    # realpath keeps a leading \\?\, and relative paths across the two forms then fail.
    if path.startswith("\\\\?\\UNC\\"):
        return "\\\\" + path[8:]
    return path[4:] if path.startswith("\\\\?\\") else path


def _resolve(path: str) -> str | None:
    try:
        return _strip_verbatim(os.path.realpath(path, strict=True))
    except (OSError, ValueError):
        return None


def _role(path: str) -> Role | None:
    try:
        tags = get_backend_system_tags_from_path(path)
    except ValueError:
        return None
    return frozenset(t for t in tags if t in _ROOT_TAGS or t.startswith("model_type:")), compute_loader_path(path)


def _same_file(old: str, new: str) -> bool:
    try:
        before, after = os.stat(old), os.stat(new)
    except (OSError, ValueError):
        return False
    return before.st_ino != 0 and (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)


def _targets(real_dir: str, name: str, folders: list[tuple[str, str]]) -> set[str]:
    """``<folder as spelled today>/<path below its real location>`` for each folder holding real_dir."""
    targets = set()
    for spelled, real in folders:
        stem = real.rstrip(os.sep) + os.sep
        if real_dir == real or real_dir.startswith(stem):
            targets.add(os.path.abspath(os.path.join(spelled, real_dir[len(stem):], name)))
    return targets


def _record_roles(session: Session, content_ids: list[str]) -> dict[str, set[Role]]:
    roles: dict[str, set[Role]] = {}
    for start in range(0, len(content_ids), _BATCH):
        records: dict[str, tuple[str, str | None, set[str]]] = {}
        rows = session.execute(
            sa.select(Asset.id, Asset.content_id, Asset.loader_path, AssetTag.tag_name)
            .outerjoin(AssetTag, AssetTag.asset_id == Asset.id)
            .where(Asset.content_id.in_(content_ids[start:start + _BATCH]))
        )
        for record_id, content_id, loader_path, tag in rows:
            tags = records.setdefault(record_id, (content_id, loader_path, set()))[2]
            if tag is not None and (tag in _ROOT_TAGS or tag.startswith("model_type:")):
                tags.add(tag)
        for content_id, loader_path, tags in records.values():
            roles.setdefault(content_id, set()).add((frozenset(tags), loader_path))
    return roles


def _decide(rows, prefixes, roles, decided, lock, progress, stop, done) -> None:
    """Worker body: each row's single fitting target proven to be the same file, else None."""
    try:
        folders = []
        for spelled in dict.fromkeys(prefixes):
            real = _resolve(spelled)
            progress.set()
            if real is not None:
                folders.append((os.path.abspath(spelled), real))
        real_dirs: dict[str, str | None] = {}
        for content_id, path in rows:
            directory, name = os.path.split(path)
            if directory not in real_dirs:
                real_dirs[directory] = _resolve(directory)
            real_dir = real_dirs[directory]
            targets = set() if real_dir is None else _targets(real_dir, name, folders)
            # The owners are the folders holding the file whose role fits its records.
            fitting = [target for target in targets if roles[content_id] == {_role(target)}]
            target = fitting[0] if len(fitting) == 1 and _same_file(path, fitting[0]) else None
            with lock:
                if stop.is_set():
                    return
                decided[content_id] = target
            progress.set()
    except Exception:
        logging.exception("re-homing pruned asset rows failed")
    finally:
        done.set()
        progress.set()


def _decide_with_stall_timeout(rows, prefixes, roles) -> dict[str, str | None]:
    decided: dict[str, str | None] = {}
    lock = threading.Lock()
    progress, stop, done = threading.Event(), threading.Event(), threading.Event()
    worker = threading.Thread(
        target=_decide,
        args=(rows, prefixes, roles, decided, lock, progress, stop, done),
        name="assets-prune-rehome",
        daemon=True,
    )
    worker.start()
    while not done.is_set():
        if not progress.wait(STALL_SECONDS):
            logging.warning("Asset prune: filesystem stalled; undecided rows are not re-homed")
            break
        progress.clear()
    with lock:
        stop.set()
        return dict(decided)


def _taken_paths(session: Session, paths: list[str]) -> set[str]:
    """Paths held by a live row, or by a missing one that recovery could still bring back."""
    taken: set[str] = set()
    has_records = sa.exists().where(Asset.content_id == AssetContent.id)
    for start in range(0, len(paths), _BATCH):
        taken.update(session.scalars(
            sa.select(AssetContent.path).where(
                AssetContent.path.in_(paths[start:start + _BATCH]),
                sa.or_(AssetContent.is_missing.is_(False), has_records),
            )
        ))
    return taken


def _text_targets(path: str, prefixes: list[str]) -> list[str]:
    """``path`` respelled by text for each folder that owns it with case folded."""
    targets = []
    for prefix in prefixes:
        base = os.path.abspath(prefix).rstrip(os.sep) + os.sep
        if os.path.normcase(path).startswith(os.path.normcase(base)):
            targets.append(base + path[len(base):])
    return targets


def plan_prune(
    session: Session, rows: list[tuple[str, str]], prefixes: list[str], spare: Callable[[str], bool] = lambda _: False
) -> PrunePlan:
    """Decide each unowned ``(content id, path)`` row: re-home, retire, or (``spare``) leave
    as it is. Reads only."""
    roles = _record_roles(session, [content_id for content_id, _ in rows])
    # A spared row whose every respelling by text is taken (its case duplicate, usually)
    # stays as it is either way; skip its filesystem reads, which would recur every boot.
    spared_targets = {cid: _text_targets(path, prefixes) for cid, path in rows if spare(path)}
    taken_by_text = _taken_paths(session, [t for targets in spared_targets.values() for t in targets])
    settled = {cid for cid, targets in spared_targets.items() if taken_by_text.issuperset(targets)}
    candidates = sorted(
        ((cid, path) for cid, path in rows if cid in roles and cid not in settled), key=lambda row: row[1]
    )
    decided = _decide_with_stall_timeout(candidates, prefixes, roles) if candidates else {}
    # Two movers for one target both retire.
    claims = Counter(decided.values())
    movers = {cid: target for cid, target in decided.items() if target is not None and claims[target] == 1}
    taken = _taken_paths(session, list(movers.values()))
    moves = {cid: target for cid, target in movers.items() if target not in taken}
    spared = frozenset(cid for cid, path in rows if spare(path))
    return PrunePlan(moves, [cid for cid, _ in rows if cid not in moves and cid not in spared], spared)


def _rewrite(session: Session, moves: list[tuple[str, str]]) -> None:
    with session.begin_nested():
        session.execute(sa.update(AssetContent), [{"id": cid, "path": path} for cid, path in moves])


def apply_prune_plan(session: Session, plan: PrunePlan) -> PruneResult:
    retire = list(plan.retire)
    moves = list(plan.moves.items())
    rehomed = 0
    for start in range(0, len(moves), _BATCH):
        batch = moves[start:start + _BATCH]
        try:
            _rewrite(session, batch)
            rehomed += len(batch)
            continue
        except IntegrityError as exc:
            if not is_live_path_conflict(exc):
                raise
        # A concurrent writer took a target since the plan read it: skip only that row.
        for move in batch:
            try:
                _rewrite(session, [move])
                rehomed += 1
            except IntegrityError as exc:
                if not is_live_path_conflict(exc):
                    raise
                if move[0] not in plan.spared:
                    retire.append(move[0])
    for content_id in retire:
        mark_content_missing(session, content_id)
    return PruneResult(len(retire), rehomed)

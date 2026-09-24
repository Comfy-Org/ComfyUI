"""Worker cleanup: ``DELETE /api/wrapper/jobs/{job_id}``.

A GPU worker should keep nothing once the caller has stored a render. After it
has (or after a failure or cancel) the caller deletes the job, and this module
removes everything the job left behind:

- output files: every file the job's ComfyUI history entry lists (``type``
  output or temp), the ``.prompt.json`` sidecar the wrapper writes next to an
  output, and whatever is left in the job's own output folder
  ``output/wrapper/<job_id>/`` (the wrapper saves there, so outputs are found
  even when the history entry is gone, e.g. after a restart);
- input files: the job's upload folder ``input/wrapper/<job_id>/``, where the
  wrapper saves every conditioning image/video/audio/reference file of the job;
- the job's history entry.

Nothing outside ComfyUI's output/input/temp directories is ever deleted: every
path is resolved to its real path first, a path whose parent directory resolves
outside its root is refused, and so is a symlink that points outside. A symlink
that stays inside is removed as a link; its target is left alone. A job still
pending or running (or whose generate request is still open) is refused with
409, and deleting an unknown or already-deleted job is a 200 with zeros.

Deliberately free of the heavy ComfyUI imports (torch, the server) so it can be
tested on its own; folder_paths is imported lazily for the real directories.
"""

import asyncio
import contextlib
import errno
import logging
import os
import stat
import threading
import uuid
from typing import NamedTuple

from aiohttp import web

from api_wrapper import progress as wrapper_progress

# The wrapper's folder under ComfyUI's input and output directories. Each job
# gets its own subfolder, named after its job id, in both.
WRAPPER_SUBDIR = "wrapper"

DELETED, MISSING, REFUSED = "deleted", "missing", "refused"


def job_subfolder(job_id):
    """'wrapper/<job_id>': where a job's uploads (under input/) and outputs
    (under output/) are saved. Also the ref prefix of its uploads."""
    return f"{WRAPPER_SUBDIR}/{job_id}"


def is_canonical_job_id(value):
    """Same rule as comfy_execution.jobs.validate_job_id: a UUID in canonical
    lowercase hyphenated form. Also what keeps a job id from being a path."""
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


class Roots(NamedTuple):
    output: str
    input: str
    temp: str


def default_roots():
    import folder_paths  # imported here: it parses the ComfyUI command line

    return Roots(folder_paths.get_output_directory(),
                 folder_paths.get_input_directory(),
                 folder_paths.get_temp_directory())


# ---------------------------------------------------------------------------
# Jobs a generate request is still working on
# ---------------------------------------------------------------------------

class JobScope:
    """One generate request's hold on its job id. ``queued`` flips to True
    once the job is on the ComfyUI queue; from then on the queue (and later
    the history) owns the job and a caller-issued DELETE cleans it up."""

    def __init__(self, job_id):
        self.job_id = job_id
        self.queued = False


_open_jobs = {}
_open_jobs_lock = threading.Lock()


@contextlib.contextmanager
def generate_scope(job_id, roots_provider=default_roots):
    """Hold ``job_id`` for the lifetime of a generate request.

    While held, DELETE answers 409: the request may still be saving uploads,
    waiting on the job, or reading its output back. If the request ends before
    the job was queued (a bad upload, a setup/validation/rewrite error, the
    client going away) the job will never run and its id was never handed out,
    so the uploads saved for it are removed right here.
    """
    scope = JobScope(job_id)
    with _open_jobs_lock:
        _open_jobs[job_id] = scope
    try:
        yield scope
    finally:
        with _open_jobs_lock:
            _open_jobs.pop(job_id, None)
        if not scope.queued:
            try:
                result = CleanupResult()
                _delete_tree(os.path.realpath(roots_provider().input),
                             os.path.join(WRAPPER_SUBDIR, job_id), "inputs", result)
                for error in result.errors:
                    logging.warning("Could not remove an upload of unqueued job %s: %s", job_id, error)
            except Exception:
                logging.exception("Could not remove the uploads of unqueued job %s", job_id)


def is_open(job_id):
    with _open_jobs_lock:
        return job_id in _open_jobs


def active_status(prompt_queue, job_id):
    """'pending' / 'in_progress' while the job cannot be cleaned, else None."""
    mutex = getattr(prompt_queue, "mutex", None) or contextlib.nullcontext()
    with mutex:
        if any(str(item[1]) == job_id for item in prompt_queue.currently_running.values()):
            return "in_progress"
        if any(str(item[1]) == job_id for item in prompt_queue.queue):
            return "pending"
    with _open_jobs_lock:
        scope = _open_jobs.get(job_id)
    if scope is not None:
        # Its generate request is still preparing it, or reading its output back.
        return "in_progress" if scope.queued else "pending"
    return None


# ---------------------------------------------------------------------------
# Deleting, safely
# ---------------------------------------------------------------------------

class CleanupResult:
    def __init__(self):
        self.outputs = 0
        self.inputs = 0
        self.history = False
        self.refused = 0
        self.errors = []

    def deleted(self):
        return {"outputs": self.outputs, "inputs": self.inputs, "history": self.history}


def _inside(root, path, allow_root=False):
    """True when real path ``path`` is ``root`` or below it (both real)."""
    try:
        common = os.path.commonpath((root, path))
    except ValueError:  # different drives, or mixed absolute/relative
        return False
    return common == root and (allow_root or path != root)


def _refuse(result, path, why):
    result.refused += 1
    logging.warning("Wrapper cleanup refused %r: %s", path, why)
    return REFUSED


def _delete_entry(root, candidate, counter, result):
    """Delete one file (or symlink) at ``candidate`` if it is inside ``root``.

    ``root`` is a real path. The parent directory is resolved first, so a
    symlinked directory anywhere on the way cannot carry the delete out of the
    root; the last component is then inspected without following it."""
    try:
        parent = os.path.realpath(os.path.dirname(candidate))
    except (ValueError, OSError):
        return _refuse(result, candidate, "unresolvable path")
    name = os.path.basename(candidate)
    if name in ("", ".", ".."):
        return _refuse(result, candidate, "not a file name")
    if not _inside(root, parent, allow_root=True):
        return _refuse(result, candidate, f"resolves outside {root}")
    path = os.path.join(parent, name)
    try:
        mode = os.lstat(path).st_mode
    except (FileNotFoundError, NotADirectoryError):
        return MISSING
    except ValueError:
        return _refuse(result, path, "unresolvable path")
    except OSError as e:
        result.errors.append(f"{path}: {e}")
        return MISSING
    if stat.S_ISLNK(mode):
        if not _inside(root, os.path.realpath(path)):
            return _refuse(result, path, f"symlink pointing outside {root}")
    elif stat.S_ISDIR(mode):
        return _refuse(result, path, "is a directory")
    try:
        os.unlink(path)  # on a symlink this removes the link, never its target
    except (FileNotFoundError, NotADirectoryError):
        return MISSING
    except OSError as e:
        result.errors.append(f"{path}: {e}")
        return MISSING
    setattr(result, counter, getattr(result, counter) + 1)
    return DELETED


def _delete_tree(root, relative_dir, counter, result):
    """Empty and remove ``root/relative_dir`` without ever following a symlink
    out of it (the directory itself or anything inside)."""
    top = os.path.join(root, relative_dir)
    try:
        parent = os.path.realpath(os.path.dirname(top))
    except (ValueError, OSError):
        _refuse(result, top, "unresolvable path")
        return
    if not _inside(root, parent, allow_root=True):
        _refuse(result, top, f"resolves outside {root}")
        return
    top = os.path.join(parent, os.path.basename(top))
    try:
        mode = os.lstat(top).st_mode
    except (FileNotFoundError, NotADirectoryError):
        return
    except OSError as e:
        result.errors.append(f"{top}: {e}")
        return
    if not stat.S_ISDIR(mode):  # a symlink or a stray file in the job's place
        _delete_entry(root, top, counter, result)
        return

    def walk_error(error):
        result.errors.append(f"{getattr(error, 'filename', top)}: {error}")

    for dirpath, dirnames, filenames in os.walk(top, topdown=False, onerror=walk_error, followlinks=False):
        for name in filenames:
            _delete_entry(root, os.path.join(dirpath, name), counter, result)
        for name in dirnames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):  # listed as a dir, but never descended into
                _delete_entry(root, path, counter, result)
            else:
                _remove_dir(path, result)
    _remove_dir(top, result)


def _remove_dir(path, result):
    try:
        os.rmdir(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        # Something refused (a symlink pointing outside) is still in there.
        if e.errno not in (errno.ENOTEMPTY, errno.EEXIST):
            result.errors.append(f"{path}: {e}")


def _history_outputs(entry):
    """(type, subfolder, filename) of every file a history entry lists, under
    any output key (images, videos, audio, 3d, ...)."""
    for node_outputs in (entry.get("outputs") or {}).values():
        if not isinstance(node_outputs, dict):
            continue
        for items in node_outputs.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename")
                subfolder = item.get("subfolder") or ""
                if isinstance(filename, str) and filename and isinstance(subfolder, str):
                    yield item.get("type") or "output", subfolder, filename


def clean_job(prompt_queue, job_id, roots):
    """Delete a finished (or failed, cancelled, unknown) job's files and its
    history entry. The caller has checked that the job is not active.

    The history entry is dropped only when every file was dealt with, so a
    retry after a transient error can still find the outputs it lists."""
    result = CleanupResult()
    output_root = os.path.realpath(roots.output)
    input_root = os.path.realpath(roots.input)
    temp_root = os.path.realpath(roots.temp)
    job_dir = os.path.join(WRAPPER_SUBDIR, job_id)
    job_input_dir = os.path.join(input_root, job_dir)

    entry = (prompt_queue.get_history(prompt_id=job_id) or {}).get(job_id)
    if entry is not None:
        for kind, subfolder, filename in _history_outputs(entry):
            if kind == "output":
                candidate = os.path.join(output_root, subfolder, filename)
                if _delete_entry(output_root, candidate, "outputs", result) != REFUSED:
                    sidecar = os.path.splitext(candidate)[0] + ".prompt.json"
                    _delete_entry(output_root, sidecar, "outputs", result)
            elif kind == "temp":
                _delete_entry(temp_root, os.path.join(temp_root, subfolder, filename), "outputs", result)
            elif kind == "input":
                # ComfyUI's input folder is shared with the UI's own uploads:
                # only this job's upload folder is the job's to delete (and it
                # is emptied below anyway).
                candidate = os.path.join(input_root, subfolder, filename)
                if not _inside(job_input_dir, os.path.normpath(candidate)):
                    _refuse(result, candidate, "input file outside this job's upload folder")
            else:
                _refuse(result, f"{kind}:{subfolder}/{filename}", "unknown output type")

    _delete_tree(output_root, job_dir, "outputs", result)
    _delete_tree(input_root, job_dir, "inputs", result)

    if entry is not None and not result.errors:
        prompt_queue.delete_history_item(job_id)
        result.history = True
    wrapper_progress.forget(job_id)
    return result


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

def _error(message, details="", status=400, **extra):
    """Same shape as the other wrapper errors (api_wrapper.routes._error_response)."""
    body = {"error": {"type": "wrapper_api_error", "message": message,
                      "details": details, "extra_info": {}},
            "missing": []}
    body.update(extra)
    return web.json_response(body, status=status)


def make_delete_job_handler(prompt_queue, roots_provider=default_roots):
    async def delete_job(request):
        job_id = request.match_info["job_id"]
        if not is_canonical_job_id(job_id):
            return _error("Invalid job id", "job_id must be a canonical UUID.", status=400)

        status = active_status(prompt_queue, job_id)
        if status is not None:
            return _error("Job still active",
                          f"Job status is '{status}'. Wait for it to finish, or cancel it "
                          f"(POST /api/jobs/{job_id}/cancel), then delete it.", status=409)

        # File IO off the event loop: a large video can take a moment to unlink.
        result = await asyncio.to_thread(clean_job, prompt_queue, job_id, roots_provider())
        if result.errors:
            logging.warning("Wrapper cleanup of job %s incomplete: %s", job_id, "; ".join(result.errors))
            return _error("Cleanup incomplete",
                          f"{len(result.errors)} file(s) could not be deleted; the history entry was "
                          "kept so a retry can finish the job.",
                          status=500, deleted=result.deleted(), refused=result.refused)
        logging.info("Wrapper cleanup of job %s: %d output(s), %d input(s), history %s, %d refused",
                     job_id, result.outputs, result.inputs, result.history, result.refused)
        return web.json_response({"deleted": result.deleted(), "refused": result.refused})

    return delete_job


def register_routes(routes, prompt_queue, roots_provider=default_roots):
    """DELETE /wrapper/jobs/{job_id}; the server mirrors it under /api."""
    routes.delete("/wrapper/jobs/{job_id}")(make_delete_job_handler(prompt_queue, roots_provider))

"""Download files with progress indicators."""

import email.message
import logging
import mimetypes
import os
from http import HTTPStatus
from typing import BinaryIO, Iterable, Optional, Tuple

from pip._vendor.requests.models import Response
from pip._vendor.urllib3.exceptions import ReadTimeoutError

from pip._internal.cli.progress_bars import get_download_progress_renderer
from pip._internal.exceptions import IncompleteDownloadError, NetworkConnectionError
from pip._internal.models.index import PyPI
from pip._internal.models.link import Link
from pip._internal.network.cache import is_from_cache
from pip._internal.network.session import PipSession
from pip._internal.network.utils import HEADERS, raise_for_status, response_chunks
from pip._internal.utils.misc import format_size, redact_auth_from_url, splitext

logger = logging.getLogger(__name__)


def _get_http_response_size(resp: Response) -> Optional[int]:
    try:
        return int(resp.headers["content-length"])
    except (ValueError, KeyError, TypeError):
        return None


def _get_http_response_etag_or_last_modified(resp: Response) -> Optional[str]:
    """
    Return either the ETag or Last-Modified header (or None if neither exists).
    The return value can be used in an If-Range header.
    """
    return resp.headers.get("etag", resp.headers.get("last-modified"))


def _prepare_download(
    resp: Response,
    link: Link,
    progress_bar: str,
    total_length: Optional[int],
    range_start: Optional[int] = 0,
) -> Iterable[bytes]:
    if link.netloc == PyPI.file_storage_domain:
        url = link.show_url
    else:
        url = link.url_without_fragment

    logged_url = redact_auth_from_url(url)

    if total_length:
        if range_start:
            logged_url = (
                f"{logged_url} ({format_size(range_start)}/{format_size(total_length)})"
            )
        else:
            logged_url = f"{logged_url} ({format_size(total_length)})"

    if is_from_cache(resp):
        logger.info("Using cached %s", logged_url)
    elif range_start:
        logger.info("Resuming download %s", logged_url)
    else:
        logger.info("Downloading %s", logged_url)

    if logger.getEffectiveLevel() > logging.INFO:
        show_progress = False
    elif is_from_cache(resp):
        show_progress = False
    elif not total_length:
        show_progress = True
    elif total_length > (512 * 1024):
        show_progress = True
    else:
        show_progress = False

    chunks = response_chunks(resp)

    if not show_progress:
        return chunks

    renderer = get_download_progress_renderer(
        bar_type=progress_bar, size=total_length, initial_progress=range_start
    )
    return renderer(chunks)


def sanitize_content_filename(filename: str) -> str:
    """
    Sanitize the "filename" value from a Content-Disposition header.
    """
    return os.path.basename(filename)


def parse_content_disposition(content_disposition: str, default_filename: str) -> str:
    """
    Parse the "filename" value from a Content-Disposition header, and
    return the default filename if the result is empty.
    """
    m = email.message.Message()
    m["content-type"] = content_disposition
    filename = m.get_param("filename")
    if filename:
        # We need to sanitize the filename to prevent directory traversal
        # in case the filename contains ".." path parts.
        filename = sanitize_content_filename(str(filename))
    return filename or default_filename


def _get_http_response_filename(resp: Response, link: Link) -> str:
    """Get an ideal filename from the given HTTP response, falling back to
    the link filename if not provided.
    """
    filename = link.filename  # fallback
    # Have a look at the Content-Disposition header for a better guess
    content_disposition = resp.headers.get("content-disposition")
    if content_disposition:
        filename = parse_content_disposition(content_disposition, filename)
    ext: Optional[str] = splitext(filename)[1]
    if not ext:
        ext = mimetypes.guess_extension(resp.headers.get("content-type", ""))
        if ext:
            filename += ext
    if not ext and link.url != resp.url:
        ext = os.path.splitext(resp.url)[1]
        if ext:
            filename += ext
    return filename


def _http_get_download(
    session: PipSession,
    link: Link,
    range_start: Optional[int] = 0,
    if_range: Optional[str] = None,
) -> Response:
    target_url = link.url.split("#", 1)[0]
    headers = HEADERS.copy()
    # request a partial download
    if range_start:
        headers["Range"] = f"bytes={range_start}-"
    # make sure the file hasn't changed
    if if_range:
        headers["If-Range"] = if_range
    try:
        resp = session.get(target_url, headers=headers, stream=True)
        raise_for_status(resp)
    except NetworkConnectionError as e:
        assert e.response is not None
        logger.critical("HTTP error %s while getting %s", e.response.status_code, link)
        raise
    return resp


class Downloader:
    def __init__(
        self,
        session: PipSession,
        progress_bar: str,
        resume_retries: int,
    ) -> None:
        assert (
            resume_retries >= 0
        ), "Number of max resume retries must be bigger or equal to zero"
        self._session = session
        self._progress_bar = progress_bar
        self._resume_retries = resume_retries

    def __call__(self, link: Link, location: str) -> Tuple[str, str]:
        """Download the file given by link into location."""
        resp = _http_get_download(self._session, link)
        # NOTE: The original download size needs to be passed down everywhere
        # so if the download is resumed (with a HTTP Range request) the progress
        # bar will report the right size.
        total_length = _get_http_response_size(resp)
        content_type = resp.headers.get("Content-Type", "")

        filename = _get_http_response_filename(resp, link)
        filepath = os.path.join(location, filename)

        with open(filepath, "wb") as content_file:
            bytes_received = self._process_response(
                resp, link, content_file, 0, total_length
            )
            # If possible, check for an incomplete download and attempt resuming.
            if total_length and bytes_received < total_length:
                self._attempt_resume(
                    resp, link, content_file, total_length, bytes_received
                )

        return filepath, content_type

    def _process_response(
        self,
        resp: Response,
        link: Link,
        content_file: BinaryIO,
        bytes_received: int,
        total_length: Optional[int],
    ) -> int:
        """Process the response and write the chunks to the file."""
        chunks = _prepare_download(
            resp, link, self._progress_bar, total_length, range_start=bytes_received
        )
        return self._write_chunks_to_file(
            chunks, content_file, allow_partial=bool(total_length)
        )

    def _write_chunks_to_file(
        self, chunks: Iterable[bytes], content_file: BinaryIO, *, allow_partial: bool
    ) -> int:
        """Write the chunks to the file and return the number of bytes received."""
        bytes_received = 0
        try:
            for chunk in chunks:
                bytes_received += len(chunk)
                content_file.write(chunk)
        except ReadTimeoutError as e:
            # If partial downloads are OK (the download will be retried), don't bail.
            if not allow_partial:
                raise e

            # Ensuring bytes_received is returned to attempt resume
            logger.warning("Connection timed out while downloading.")

        return bytes_received

    def _attempt_resume(
        self,
        resp: Response,
        link: Link,
        content_file: BinaryIO,
        total_length: Optional[int],
        bytes_received: int,
    ) -> None:
        """Attempt to resume the download if connection was dropped."""
        etag_or_last_modified = _get_http_response_etag_or_last_modified(resp)

        attempts_left = self._resume_retries
        while total_length and attempts_left and bytes_received < total_length:
            attempts_left -= 1

            logger.warning(
                "Attempting to resume incomplete download (%s/%s, attempt %d)",
                format_size(bytes_received),
                format_size(total_length),
                (self._resume_retries - attempts_left),
            )

            try:
                # Try to resume the download using a HTTP range request.
                resume_resp = _http_get_download(
                    self._session,
                    link,
                    range_start=bytes_received,
                    if_range=etag_or_last_modified,
                )

                # Fallback: if the server responded with 200 (i.e., the file has
                # since been modified or range requests are unsupported) or any
                # other unexpected status, restart the download from the beginning.
                must_restart = resume_resp.status_code != HTTPStatus.PARTIAL_CONTENT
                if must_restart:
                    bytes_received, total_length, etag_or_last_modified = (
                        self._reset_download_state(resume_resp, content_file)
                    )

                bytes_received += self._process_response(
                    resume_resp, link, content_file, bytes_received, total_length
                )
            except (ConnectionError, ReadTimeoutError, OSError):
                continue

        # No more resume attempts. Raise an error if the download is still incomplete.
        if total_length and bytes_received < total_length:
            os.remove(content_file.name)
            raise IncompleteDownloadError(
                link, bytes_received, total_length, retries=self._resume_retries
            )

    def _reset_download_state(
        self,
        resp: Response,
        content_file: BinaryIO,
    ) -> Tuple[int, Optional[int], Optional[str]]:
        """Reset the download state to restart downloading from the beginning."""
        content_file.seek(0)
        content_file.truncate()
        bytes_received = 0
        total_length = _get_http_response_size(resp)
        etag_or_last_modified = _get_http_response_etag_or_last_modified(resp)

        return bytes_received, total_length, etag_or_last_modified


class BatchDownloader:
    def __init__(
        self,
        session: PipSession,
        progress_bar: str,
        resume_retries: int,
    ) -> None:
        self._downloader = Downloader(session, progress_bar, resume_retries)

    def __call__(
        self, links: Iterable[Link], location: str
    ) -> Iterable[Tuple[Link, Tuple[str, str]]]:
        """Download the files given by links into location."""
        for link in links:
            filepath, content_type = self._downloader(link, location)
            yield link, (filepath, content_type)

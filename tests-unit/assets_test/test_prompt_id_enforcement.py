"""POST /prompt enforces canonical-UUID job ids at creation time.

Lives in assets_test because it uses this suite's booted-server fixture. The
invariant itself is pipeline-wide: a job id is stored and compared verbatim
downstream — history keys, websocket correlation, and /interrupt matching —
so a job minted with a non-canonical id would miss every exact-match lookup.

The prompt bodies here are intentionally invalid workflows — prompt_id
validation happens before workflow validation, so a rejected id returns
``invalid_prompt_id`` while an accepted id falls through to the ordinary
workflow-validation error (proving it cleared the id check).
"""
import requests


def _post_prompt(http: requests.Session, api_base: str, body: dict) -> requests.Response:
    return http.post(api_base + "/prompt", json=body, timeout=30)


def _error_type(r: requests.Response) -> str:
    return r.json()["error"]["type"]


def test_non_uuid_prompt_id_rejected(http: requests.Session, api_base: str):
    r = _post_prompt(http, api_base, {"prompt": {}, "prompt_id": "not-a-uuid"})
    assert r.status_code == 400, r.text
    assert _error_type(r) == "invalid_prompt_id"


def test_non_string_prompt_id_rejected(http: requests.Session, api_base: str):
    # Previously str()-coerced (123 became the job id "123"); must now be a 400,
    # not a 500 from uuid.UUID choking on a non-string.
    r = _post_prompt(http, api_base, {"prompt": {}, "prompt_id": 123})
    assert r.status_code == 400, r.text
    assert _error_type(r) == "invalid_prompt_id"


def test_non_canonical_uuid_rejected(http: requests.Session, api_base: str):
    # Parseable as a UUID, but not the canonical lowercase form: rejected
    # loudly rather than silently rewritten (downstream lookups match the
    # stored id exactly).
    r = _post_prompt(
        http,
        api_base,
        {"prompt": {}, "prompt_id": "AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE"},
    )
    assert r.status_code == 400, r.text
    assert _error_type(r) == "invalid_prompt_id"


def test_canonical_uuid_accepted(http: requests.Session, api_base: str):
    # The id clears validation; the empty workflow then fails ordinary prompt
    # validation, proving the request got past the id check.
    r = _post_prompt(
        http,
        api_base,
        {"prompt": {}, "prompt_id": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"},
    )
    assert r.status_code == 400, r.text
    assert _error_type(r) != "invalid_prompt_id"


def test_null_prompt_id_not_rejected(http: requests.Session, api_base: str):
    # Explicit null means "server generates" and must not be rejected as an
    # invalid id. (The minted id itself is not observable here because the
    # workflow is invalid; unit tests cover validate_job_id directly.)
    r = _post_prompt(http, api_base, {"prompt": {}, "prompt_id": None})
    assert r.status_code == 400, r.text
    assert _error_type(r) != "invalid_prompt_id"

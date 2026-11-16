"""The vendored openapi-v2.yaml is generated, leak-free and the reviewed contract."""

from .contract import HTTP_METHODS, REVIEWED_NORMATIVE_DIGEST, SPEC, SPEC_PATH, normative_digest, resolve

EXPECTED_OPERATION_IDS = {
    "postAssets", "assetFromHash", "headAssetByHash", "getAsset", "deleteAsset", "getAssetContent",
    "postJobs", "listJobs", "getJob", "getJobWorkflow", "getJobLogs", "getJobEvents", "cancelJob",
    "getHealth", "assetFromPath",
}


def _operations():
    return {op["operationId"]: op for item in SPEC["paths"].values() for m, op in item.items() if m in HTTP_METHODS}


def _refs(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "$ref":
                yield v
            else:
                yield from _refs(v)
    elif isinstance(node, list):
        for v in node:
            yield from _refs(v)


def test_file_is_generated():
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert text.startswith("# Comfy API v2 — public specification.\n")
    assert "GENERATED ONE-WAY — DO NOT HAND-EDIT" in text


def test_openapi_and_major_version():
    assert SPEC["openapi"] == "3.0.3"
    assert str(SPEC["info"]["version"]).startswith("2.")


def test_no_internal_operations():
    for op_id, op in _operations().items():
        assert "internal" not in op.get("tags", []), op_id
        assert op.get("x-internal") is not True, op_id


def test_operation_ids_are_exactly_the_contract():
    assert set(_operations()) == EXPECTED_OPERATION_IDS


def test_from_path_and_health_responses():
    ops = _operations()
    assert "501" in ops["assetFromPath"]["responses"]
    health = ops["getHealth"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert health == {"$ref": "#/components/schemas/Health"}


def test_every_ref_resolves():
    refs = set(_refs(SPEC))
    refs |= {e["schema"] for e in _operations()["getJobEvents"]["x-sse-events"].values()}
    assert refs
    for ref in refs:
        resolve(ref)


def test_sse_event_catalog():
    events = _operations()["getJobEvents"]["x-sse-events"]
    assert {"status", "progress", "preview", "output", "log", "error"} == set(events)


def test_normative_digest_is_reviewed():
    digest = normative_digest(SPEC)
    assert digest == REVIEWED_NORMATIVE_DIGEST, (
        "openapi-v2.yaml changed normatively (requests, parameters, responses, headers, schemas or SSE). "
        "Review the diff, update app/api_v2 and its tests for the change, "
        f"then paste the new digest {digest} into contract.py."
    )

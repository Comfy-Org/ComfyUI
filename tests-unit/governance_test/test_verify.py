import base64
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from app import governance


DOMAIN_SEPARATOR = b"comfyui-governance-v1\x00"
BASE64URL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
PAYLOAD_KEYS = {
    "schemaVersion",
    "audience",
    "versionId",
    "buildIdentity",
    "sourcesDigest",
    "policyGeneration",
    "activeForms",
    "customNodeMode",
    "packs",
    "deniedPacks",
    "disabledNodes",
    "disabledPartnerNodes",
    "models",
}


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _payload() -> dict:
    return {
        "schemaVersion": 1,
        "audience": "comfyui-core",
        "versionId": "version-1",
        "buildIdentity": "test-build",
        "sourcesDigest": "sources-digest",
        "policyGeneration": 7,
        "activeForms": ["customNode"],
        "customNodeMode": "allowlist",
        "packs": [],
        "deniedPacks": [],
        "disabledNodes": [],
        "disabledPartnerNodes": [],
        "models": [],
    }


def _envelope(
    private_key: Ed25519PrivateKey,
    payload: dict | None = None,
    *,
    payload_bytes: bytes | None = None,
    domain_separated: bool = True,
) -> bytes:
    if payload_bytes is None:
        payload_bytes = json.dumps(payload if payload is not None else _payload(), separators=(",", ":")).encode()
    message = DOMAIN_SEPARATOR + payload_bytes if domain_separated else payload_bytes
    outer = {"schema": 1, "payload": _b64url(payload_bytes), "signature": _b64url(private_key.sign(message))}
    return json.dumps(outer, separators=(",", ":")).encode()


def _outer(envelope: bytes) -> dict:
    return json.loads(envelope)


def _encoded_outer(outer: dict) -> bytes:
    return json.dumps(outer, separators=(",", ":")).encode()


def _noncanonical_trailing_bits(encoded: str) -> str:
    assert len(encoded) % 4 in {2, 3}
    index = BASE64URL_ALPHABET.index(encoded[-1])
    return encoded[:-1] + BASE64URL_ALPHABET[index + 1]


@pytest.fixture
def private_key(monkeypatch: pytest.MonkeyPatch) -> Ed25519PrivateKey:
    key = Ed25519PrivateKey.generate()
    public_bytes = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    monkeypatch.setattr(governance, "GOVERNANCE_BUILD_IDENTITY", "test-build")
    monkeypatch.setattr(governance, "GOVERNANCE_PUBLIC_KEY", _b64url(public_bytes))
    monkeypatch.setattr(governance, "GOVERNANCE_MIN_POLICY_GENERATION", 7)
    return key


def test_valid_envelope_returns_parsed_payload(private_key: Ed25519PrivateKey) -> None:
    payload = _payload()

    assert governance.verify_and_load(_envelope(private_key, payload)) == payload


@pytest.mark.parametrize(
    ("active_forms", "custom_node_mode"),
    [(["customNode"], "allowlist"), (["model"], None)],
)
def test_active_allowlist_form_can_have_empty_resolved_data(
    private_key: Ed25519PrivateKey,
    active_forms: list[str],
    custom_node_mode: str | None,
) -> None:
    payload = _payload()
    payload.update(activeForms=active_forms, customNodeMode=custom_node_mode)

    loaded = governance.verify_and_load(_envelope(private_key, payload))

    assert loaded["activeForms"] == active_forms


def test_payload_tampering_fails_signature_verification(private_key: Ed25519PrivateKey) -> None:
    outer = _outer(_envelope(private_key))
    payload_bytes = base64.urlsafe_b64decode(outer["payload"] + "==")
    outer["payload"] = _b64url(payload_bytes.replace(b"comfyui-core", b"comfyui-corf"))

    with pytest.raises(InvalidSignature):
        governance.verify_and_load(_encoded_outer(outer))


def test_signature_tampering_fails_verification(private_key: Ed25519PrivateKey) -> None:
    outer = _outer(_envelope(private_key))
    signature = bytearray(base64.urlsafe_b64decode(outer["signature"] + "=="))
    signature[0] ^= 1
    outer["signature"] = _b64url(bytes(signature))

    with pytest.raises(InvalidSignature):
        governance.verify_and_load(_encoded_outer(outer))


def test_signature_from_wrong_key_is_rejected(private_key: Ed25519PrivateKey) -> None:
    with pytest.raises(InvalidSignature):
        governance.verify_and_load(_envelope(Ed25519PrivateKey.generate()))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("audience", "desktop"),
        ("buildIdentity", "foreign-build"),
        ("schemaVersion", 2),
    ],
)
def test_signed_payload_identity_mismatches_are_rejected(
    private_key: Ed25519PrivateKey,
    key: str,
    value: str | int,
) -> None:
    payload = _payload()
    payload[key] = value

    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key, payload))


def test_signature_without_domain_separator_is_rejected(private_key: Ed25519PrivateKey) -> None:
    with pytest.raises(InvalidSignature):
        governance.verify_and_load(_envelope(private_key, domain_separated=False))


def test_policy_generation_below_floor_is_rejected(private_key: Ed25519PrivateKey) -> None:
    payload = _payload()
    payload["policyGeneration"] = 6

    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key, payload))


def test_policy_generation_equal_to_floor_is_accepted(private_key: Ed25519PrivateKey) -> None:
    assert governance.verify_and_load(_envelope(private_key))["policyGeneration"] == 7


@pytest.mark.parametrize("schema", [2, "1", True, None])
def test_envelope_schema_must_be_integer_one(private_key: Ed25519PrivateKey, schema) -> None:
    outer = _outer(_envelope(private_key))
    outer["schema"] = schema

    with pytest.raises(ValueError):
        governance.verify_and_load(_encoded_outer(outer))


@pytest.mark.parametrize("missing_key", ["schema", "payload", "signature"])
def test_envelope_rejects_each_missing_key(private_key: Ed25519PrivateKey, missing_key: str) -> None:
    outer = _outer(_envelope(private_key))
    del outer[missing_key]

    with pytest.raises(ValueError):
        governance.verify_and_load(_encoded_outer(outer))


def test_envelope_rejects_extra_key(private_key: Ed25519PrivateKey) -> None:
    outer = _outer(_envelope(private_key))
    outer["kid"] = "unexpected"

    with pytest.raises(ValueError):
        governance.verify_and_load(_encoded_outer(outer))


@pytest.mark.parametrize(("key", "value"), [("payload", 1), ("signature", None)])
def test_envelope_rejects_wrong_codec_field_types(
    private_key: Ed25519PrivateKey,
    key: str,
    value,
) -> None:
    outer = _outer(_envelope(private_key))
    outer[key] = value

    with pytest.raises(ValueError):
        governance.verify_and_load(_encoded_outer(outer))


@pytest.mark.parametrize("invalid_envelope", [b"not json", b"[]", b"{}", json.dumps(_payload()).encode()])
def test_unsigned_or_non_object_envelopes_are_rejected(invalid_envelope: bytes) -> None:
    with pytest.raises(ValueError):
        governance.verify_and_load(invalid_envelope)


def test_signed_payload_must_be_an_object(private_key: Ed25519PrivateKey) -> None:
    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key, payload_bytes=b"[]"))


@pytest.mark.parametrize("missing_key", sorted(PAYLOAD_KEYS))
def test_payload_rejects_each_missing_key(private_key: Ed25519PrivateKey, missing_key: str) -> None:
    payload = _payload()
    del payload[missing_key]

    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key, payload))


def test_payload_rejects_extra_key(private_key: Ed25519PrivateKey) -> None:
    payload = _payload()
    payload["modelPolicy"] = {}

    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key, payload))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schemaVersion", "1"),
        ("schemaVersion", True),
        ("audience", 1),
        ("versionId", 1),
        ("buildIdentity", 1),
        ("sourcesDigest", 1),
        ("policyGeneration", "7"),
        ("policyGeneration", True),
        ("activeForms", "customNode"),
        ("customNodeMode", 1),
        ("packs", {}),
        ("deniedPacks", {}),
        ("disabledNodes", {}),
        ("disabledPartnerNodes", {}),
        ("models", {}),
    ],
)
def test_payload_rejects_wrong_top_level_types(
    private_key: Ed25519PrivateKey,
    key: str,
    value,
) -> None:
    payload = _payload()
    payload[key] = value

    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key, payload))


@pytest.mark.parametrize(
    ("active_forms", "custom_node_mode"),
    [
        (["unknown"], None),
        (["model", "customNode"], "allowlist"),
        (["customNode", "customNode"], "allowlist"),
        ([], None),
        (["model"], "allowlist"),
        (["customNode"], None),
        (["customNode"], "audit"),
        ([1], None),
    ],
)
def test_payload_rejects_invalid_active_form_and_mode_combinations(
    private_key: Ed25519PrivateKey,
    active_forms: list,
    custom_node_mode,
) -> None:
    payload = _payload()
    payload.update(activeForms=active_forms, customNodeMode=custom_node_mode)

    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key, payload))


@pytest.mark.parametrize("denied_packs", [["pack"], ["Pack"], ["path/pack"], ["path\\pack"], ["b", "a"], ["pack", "pack"], [1]])
def test_payload_rejects_invalid_denied_packs(
    private_key: Ed25519PrivateKey,
    denied_packs: list,
) -> None:
    payload = _payload()
    payload["deniedPacks"] = denied_packs
    if denied_packs == ["pack"]:
        payload["customNodeMode"] = "allowlist"
    else:
        payload["customNodeMode"] = "blocklist"

    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key, payload))


@pytest.mark.parametrize(
    ("active_forms", "custom_node_mode", "key", "value"),
    [
        (["model"], None, "packs", [{"name": "pack", "digest": "digest"}]),
        (["customNode"], "allowlist", "disabledNodes", ["Node"]),
        (["customNode"], "allowlist", "disabledPartnerNodes", [{"nodeId": "Node", "providerId": "provider"}]),
        (["customNode"], "allowlist", "models", ["blake3:" + "a" * 64]),
    ],
)
def test_payload_rejects_nonempty_data_for_inactive_form(
    private_key: Ed25519PrivateKey,
    active_forms: list[str],
    custom_node_mode: str | None,
    key: str,
    value: list,
) -> None:
    payload = _payload()
    payload.update(activeForms=active_forms, customNodeMode=custom_node_mode)
    payload[key] = value

    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key, payload))


def test_node_id_form_rejects_empty_disabled_nodes(private_key: Ed25519PrivateKey) -> None:
    payload = _payload()
    payload.update(activeForms=["nodeId"], customNodeMode=None)

    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key, payload))


@pytest.mark.parametrize("disabled_nodes", [["B", "A"], ["A", "A"], [1]])
def test_disabled_nodes_must_be_a_sorted_unique_string_list(
    private_key: Ed25519PrivateKey,
    disabled_nodes: list,
) -> None:
    payload = _payload()
    payload.update(activeForms=["nodeId"], customNodeMode=None, disabledNodes=disabled_nodes)

    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key, payload))


@pytest.mark.parametrize(
    "packs",
    [
        ["pack"],
        [{"name": "pack"}],
        [{"name": "pack", "digest": "digest", "extra": "value"}],
        [{"name": 1, "digest": "digest"}],
        [{"name": "pack", "digest": 1}],
    ],
)
def test_packs_require_exact_string_entry_schema(private_key: Ed25519PrivateKey, packs: list) -> None:
    payload = _payload()
    payload["packs"] = packs

    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key, payload))


@pytest.mark.parametrize(
    "entries",
    [
        ["Node"],
        [{"nodeId": "Node"}],
        [{"nodeId": "Node", "providerId": "provider", "extra": "value"}],
        [{"nodeId": 1, "providerId": "provider"}],
        [{"nodeId": "Node", "providerId": 1}],
        [{"nodeId": "A", "providerId": "Provider"}],
        [{"nodeId": "A", "providerId": "provider/id"}],
        [{"nodeId": "A", "providerId": ""}],
        [{"nodeId": "B", "providerId": "b"}, {"nodeId": "A", "providerId": "a"}],
        [{"nodeId": "A", "providerId": "a"}, {"nodeId": "A", "providerId": "b"}],
    ],
)
def test_partner_nodes_require_exact_canonical_entries(private_key: Ed25519PrivateKey, entries: list) -> None:
    payload = _payload()
    payload.update(activeForms=["partnerNode"], customNodeMode=None, disabledPartnerNodes=entries)

    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key, payload))


@pytest.mark.parametrize(
    "models",
    [
        ["a" * 64],
        ["blake3:" + "A" * 64],
        ["blake3:" + "a" * 63],
        ["blake3:" + "g" * 64],
        ["blake3:" + "b" * 64, "blake3:" + "a" * 64],
        ["blake3:" + "a" * 64, "blake3:" + "a" * 64],
        [1],
    ],
)
def test_models_require_sorted_unique_canonical_blake3_digests(
    private_key: Ed25519PrivateKey,
    models: list,
) -> None:
    payload = _payload()
    payload.update(activeForms=["model"], customNodeMode=None, models=models)

    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key, payload))


@pytest.mark.parametrize("field", ["payload", "signature", "public_key"])
def test_padded_base64_is_rejected(private_key: Ed25519PrivateKey, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    outer = _outer(_envelope(private_key))
    if field == "public_key":
        monkeypatch.setattr(governance, "GOVERNANCE_PUBLIC_KEY", governance.GOVERNANCE_PUBLIC_KEY + "=")
    else:
        outer[field] += "="

    with pytest.raises(ValueError):
        governance.verify_and_load(_encoded_outer(outer))


@pytest.mark.parametrize("field", ["payload", "signature", "public_key"])
@pytest.mark.parametrize("character", ["+", "/"])
def test_non_url_safe_base64_alphabet_is_rejected(
    private_key: Ed25519PrivateKey,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    character: str,
) -> None:
    outer = _outer(_envelope(private_key))
    if field == "public_key":
        monkeypatch.setattr(governance, "GOVERNANCE_PUBLIC_KEY", character + governance.GOVERNANCE_PUBLIC_KEY[1:])
    else:
        outer[field] = character + outer[field][1:]

    with pytest.raises(ValueError):
        governance.verify_and_load(_encoded_outer(outer))


@pytest.mark.parametrize("field", ["payload", "signature", "public_key"])
def test_noncanonical_trailing_bits_are_rejected(
    private_key: Ed25519PrivateKey,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    payload_bytes = json.dumps(_payload(), separators=(",", ":")).encode()
    while len(payload_bytes) % 3 == 0:
        payload_bytes += b" "
    outer = _outer(_envelope(private_key, payload_bytes=payload_bytes))
    if field == "public_key":
        monkeypatch.setattr(governance, "GOVERNANCE_PUBLIC_KEY", _noncanonical_trailing_bits(governance.GOVERNANCE_PUBLIC_KEY))
    else:
        outer[field] = _noncanonical_trailing_bits(outer[field])

    with pytest.raises(ValueError):
        governance.verify_and_load(_encoded_outer(outer))


@pytest.mark.parametrize("signature_size", [63, 65])
def test_signature_must_decode_to_64_bytes(private_key: Ed25519PrivateKey, signature_size: int) -> None:
    outer = _outer(_envelope(private_key))
    outer["signature"] = _b64url(b"s" * signature_size)

    with pytest.raises(ValueError):
        governance.verify_and_load(_encoded_outer(outer))


@pytest.mark.parametrize("public_key_size", [31, 33])
def test_public_key_must_decode_to_32_bytes(
    private_key: Ed25519PrivateKey,
    monkeypatch: pytest.MonkeyPatch,
    public_key_size: int,
) -> None:
    monkeypatch.setattr(governance, "GOVERNANCE_PUBLIC_KEY", _b64url(b"k" * public_key_size))

    with pytest.raises(ValueError):
        governance.verify_and_load(_envelope(private_key))


def test_oversized_envelope_is_rejected_before_parsing(
    private_key: Ed25519PrivateKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = _envelope(private_key)
    monkeypatch.setattr(governance, "_MAX_ENVELOPE_BYTES", len(envelope) - 1, raising=False)
    monkeypatch.setattr(governance.json, "loads", lambda *_args, **_kwargs: pytest.fail("parsed oversized envelope"))

    with pytest.raises(ValueError):
        governance.verify_and_load(envelope)


def test_oversized_payload_is_rejected_before_decoding(
    private_key: Ed25519PrivateKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = _outer(_envelope(private_key))
    monkeypatch.setattr(governance, "_MAX_PAYLOAD_BYTES", 1, raising=False)
    monkeypatch.setattr(governance.base64, "b64decode", lambda *_args, **_kwargs: pytest.fail("decoded oversized payload"))

    with pytest.raises(ValueError):
        governance.verify_and_load(_encoded_outer(outer))


def test_valid_signature_is_checked_before_invalid_json_is_parsed(private_key: Ed25519PrivateKey) -> None:
    with pytest.raises(json.JSONDecodeError):
        governance.verify_and_load(_envelope(private_key, payload_bytes=b"not json"))


def test_invalid_signature_precedes_parsing_valid_json(private_key: Ed25519PrivateKey) -> None:
    outer = _outer(_envelope(private_key))
    outer["signature"] = _b64url(b"s" * 64)

    with pytest.raises(InvalidSignature):
        governance.verify_and_load(_encoded_outer(outer))

import base64
import binascii
import json
import logging
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Awaitable, Callable, Mapping
import unicodedata

import yaml


GOVERNANCE_REQUIRED = False
GOVERNANCE_BUILD_IDENTITY = ""
GOVERNANCE_PUBLIC_KEY = ""
GOVERNANCE_MIN_POLICY_GENERATION = 0
GOVERNANCE_CAPABILITY_VERSION = 1

_DOMAIN_SEPARATOR = b"comfyui-governance-v1\x00"
_MAX_ENVELOPE_BYTES = 1024 * 1024
_MAX_PAYLOAD_BYTES = 512 * 1024
_ENVELOPE_KEYS = {"schema", "payload", "signature"}
_PAYLOAD_KEYS = {
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
_ACTIVE_FORMS = ("customNode", "nodeId", "model", "partnerNode")
# Only add a form here once this build enforces it; anything absent is refused rather than silently allowed.
_ENFORCED_FORMS = frozenset({"customNode", "nodeId", "partnerNode"})
_BASE64URL_PATTERN = re.compile(r"[A-Za-z0-9_-]*")
_PROVIDER_ID_PATTERN = re.compile(r"[a-z0-9._-]+")
_MODEL_DIGEST_PATTERN = re.compile(r"blake3:[0-9a-f]{64}")
_PACK_EXTENSIONS = frozenset({".py", ".pyd", ".so", ".dll", ".dylib"})
_BYTECODE_EXTENSIONS = frozenset({".pyc", ".pyo"})

_COMFYUI_ROOT = Path(__file__).parent.parent
_POLICY_PATH = _COMFYUI_ROOT / "governance" / "policy.signed.json"
_EXTRA_MODEL_PATHS_CONFIG_PATH = _COMFYUI_ROOT / "extra_model_paths.yaml"
_policy: dict | None = None
_disabled_nodes: frozenset[str] = frozenset()
_original_load_custom_node: Callable[[str, set[str], str], Awaitable[bool]] | None = None
_custom_node_mode: str | None = None
_denied_packs: frozenset[str] = frozenset()
_allowed_packs: Mapping[str, str] = MappingProxyType({})
# Message attribution only. Enforcement continues to use _disabled_nodes.
_partner_provider_map: Mapping[str, str] = MappingProxyType({})


def load_disabled_nodes(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("disabled nodes config must be a mapping")
    if "disabled_nodes" not in config:
        raise ValueError("disabled nodes config must contain disabled_nodes")

    disabled_nodes = config["disabled_nodes"]
    if not isinstance(disabled_nodes, list):
        raise ValueError("disabled_nodes must be a list")
    if not all(isinstance(node_id, str) for node_id in disabled_nodes):
        raise ValueError("disabled_nodes entries must be strings")

    return set(disabled_nodes)


def set_partner_provider_map(mapping: dict[str, str]) -> None:
    global _partner_provider_map
    _partner_provider_map = MappingProxyType(mapping.copy())


def partner_provider_for_node(node_id: str) -> str | None:
    return _partner_provider_map.get(node_id)


def node_policy_message(node_id: str) -> str | None:
    provider_id = partner_provider_for_node(node_id)
    if provider_id is not None:
        return f"Provider '{provider_id}' is not permitted by your organization's policy."
    if node_id in _disabled_nodes:
        return f"Node '{node_id}' is not permitted by your organization's policy."
    return None


def set_custom_node_policy(mode: str | None, denied_packs: frozenset[str], allowed_packs: dict[str, str]) -> None:
    if mode is not None and mode not in {"allowlist", "blocklist"}:
        raise ValueError("custom-node policy mode must be allowlist, blocklist, or None")

    global _custom_node_mode, _denied_packs, _allowed_packs
    _custom_node_mode = mode
    _denied_packs = frozenset(denied_packs)
    _allowed_packs = MappingProxyType({name.lower(): digest for name, digest in allowed_packs.items()})
    if mode is not None:
        # Cached bytecode runs without reading the source the digest measures, so a gated install must never write any.
        sys.dont_write_bytecode = True


def pack_allowed(module_path: str) -> bool:
    if _custom_node_mode is None:
        return True

    basename = Path(module_path).name
    expected_digest = _allowed_packs.get(basename.lower())
    if _custom_node_mode == "blocklist":
        if basename.lower() in _denied_packs:
            return False
        if expected_digest is None:
            return True

    try:
        digest = pack_digest(module_path)
    except ValueError as error:
        logging.warning("Cannot verify custom node pack '%s': %s", basename, error)
        return False

    if _custom_node_mode == "allowlist":
        return digest in _allowed_packs.values()
    return digest == expected_digest


def pack_digest(pack_path: str) -> str:
    # Keep local: hashing imports comfy.cli_args, which would freeze CLI defaults before main enables argument parsing.
    from app.assets.services.hashing import compute_blake3_hash
    from blake3 import blake3

    path = Path(pack_path)
    if path.is_symlink():
        raise ValueError("pack path must not be a symlink")

    if path.is_file():
        if path.suffix.lower() != ".py":
            raise ValueError("single-file packs must be Python files")
        root, candidates = path.parent, [path]
    elif path.is_dir():
        root, candidates = path, list(path.rglob("*"))
    else:
        raise ValueError("pack path must be a directory or Python file")

    if any(candidate.is_symlink() for candidate in candidates):
        raise ValueError("pack must not contain symlinks")
    contents = [(candidate.relative_to(root), candidate) for candidate in candidates if candidate.is_file()]
    # An unchecked hash-based .pyc executes without reading the .py this digest measures, so bytecode is refused outright.
    if any(relative.suffix.lower() in _BYTECODE_EXTENSIONS or "__pycache__" in relative.parts for relative, _ in contents):
        raise ValueError("pack must not contain compiled Python bytecode")
    files = [
        (unicodedata.normalize("NFC", relative.as_posix()), candidate)
        for relative, candidate in contents
        if relative.suffix.lower() in _PACK_EXTENSIONS
    ]
    if len(files) != len({relative_path for relative_path, _ in files}):
        raise ValueError("pack paths must remain unique after NFC normalization")

    hasher = blake3()
    for relative_path, file_path in sorted(files, key=lambda record: record[0].encode("utf-8")):
        file_digest, _ = compute_blake3_hash(str(file_path))
        assert file_digest is not None
        hasher.update(relative_path.encode("utf-8") + b"\x00" + bytes.fromhex(file_digest) + b"\x00")
    return "blake3:" + hasher.hexdigest()


def apply_disabled_nodes(disabled: set[str]) -> None:
    global _disabled_nodes, _original_load_custom_node

    disabled = disabled.union(_disabled_nodes)
    if not disabled:
        return

    import nodes

    missing = disabled.difference(nodes.NODE_CLASS_MAPPINGS)
    pruned = len(disabled) - len(missing)
    for node_id in disabled:
        nodes.NODE_CLASS_MAPPINGS.pop(node_id, None)
        nodes.NODE_DISPLAY_NAME_MAPPINGS.pop(node_id, None)

    _disabled_nodes = _disabled_nodes.union(disabled)
    if _original_load_custom_node is None:
        _original_load_custom_node = nodes.load_custom_node
        original_load_custom_node = _original_load_custom_node

        async def load_custom_node(module_path: str, ignore: set[str] = set(), module_parent: str = "custom_nodes") -> bool:
            return await original_load_custom_node(module_path, ignore | _disabled_nodes, module_parent)

        nodes.load_custom_node = load_custom_node

    if missing:
        logging.warning("Disabled node IDs were not registered: %s", ", ".join(sorted(missing)))
    logging.info("Pruned %d disabled node%s.", pruned, "" if pruned == 1 else "s")


def verify_and_load(envelope_bytes: bytes) -> dict:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if len(envelope_bytes) > _MAX_ENVELOPE_BYTES:
        raise ValueError("governance envelope exceeds maximum size")

    envelope = json.loads(envelope_bytes)
    if not isinstance(envelope, dict) or set(envelope) != _ENVELOPE_KEYS:
        raise ValueError("governance envelope must contain exactly schema, payload, and signature")
    if type(envelope["schema"]) is not int or envelope["schema"] != 1:
        raise ValueError("unsupported governance envelope schema")
    if not isinstance(envelope["payload"], str) or not isinstance(envelope["signature"], str):
        raise ValueError("governance payload and signature must be strings")

    max_encoded_payload_size = (_MAX_PAYLOAD_BYTES * 4 + 2) // 3
    if len(envelope["payload"]) > max_encoded_payload_size:
        raise ValueError("governance payload exceeds maximum size")

    payload_bytes = _decode_base64url(envelope["payload"], "payload")
    if len(payload_bytes) > _MAX_PAYLOAD_BYTES:
        raise ValueError("governance payload exceeds maximum size")
    signature = _decode_base64url(envelope["signature"], "signature")
    if len(signature) != 64:
        raise ValueError("governance signature must be 64 bytes")
    public_key_bytes = _decode_base64url(GOVERNANCE_PUBLIC_KEY, "public key")
    if len(public_key_bytes) != 32:
        raise ValueError("governance public key must be 32 bytes")

    Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(signature, _DOMAIN_SEPARATOR + payload_bytes)

    payload = json.loads(payload_bytes)
    _validate_payload(payload)
    return payload


def _decode_base64url(encoded: str, field: str) -> bytes:
    if not isinstance(encoded, str) or _BASE64URL_PATTERN.fullmatch(encoded) is None:
        raise ValueError(f"governance {field} must use unpadded base64url")

    try:
        decoded = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"governance {field} is not valid base64url") from error

    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    if canonical != encoded:
        raise ValueError(f"governance {field} is not canonically encoded")
    return decoded


def _validate_payload(payload: dict) -> None:
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
        raise ValueError("governance payload has an unexpected schema")

    if type(payload["schemaVersion"]) is not int or payload["schemaVersion"] != 1:
        raise ValueError("unsupported governance payload schema")
    for key in ("audience", "versionId", "buildIdentity", "sourcesDigest"):
        if not isinstance(payload[key], str):
            raise ValueError(f"governance payload {key} must be a string")
    if payload["audience"] != "comfyui-core":
        raise ValueError("governance payload has the wrong audience")
    if payload["buildIdentity"] != GOVERNANCE_BUILD_IDENTITY:
        raise ValueError("governance payload has the wrong build identity")
    if type(payload["policyGeneration"]) is not int:
        raise ValueError("governance policy generation must be an integer")
    if payload["policyGeneration"] < GOVERNANCE_MIN_POLICY_GENERATION:
        raise ValueError("governance policy generation is below the required minimum")

    active_forms = payload["activeForms"]
    if not isinstance(active_forms, list) or not all(isinstance(form, str) for form in active_forms):
        raise ValueError("governance activeForms must be a string list")
    if not active_forms:
        raise ValueError("governance activeForms must not be empty")
    if active_forms != [form for form in _ACTIVE_FORMS if form in active_forms]:
        raise ValueError("governance activeForms must be unique and canonically ordered")

    custom_node_mode = payload["customNodeMode"]
    custom_node_active = "customNode" in active_forms
    if custom_node_active and custom_node_mode not in {"allowlist", "blocklist"}:
        raise ValueError("governance customNodeMode is invalid for an active customNode form")
    if not custom_node_active and custom_node_mode is not None:
        raise ValueError("governance customNodeMode must be null when customNode is inactive")

    packs = payload["packs"]
    if not isinstance(packs, list):
        raise ValueError("governance packs must be a list")
    for pack in packs:
        if not isinstance(pack, dict) or set(pack) != {"name", "digest"}:
            raise ValueError("governance pack entries must contain exactly name and digest")
        if not isinstance(pack["name"], str) or not isinstance(pack["digest"], str):
            raise ValueError("governance pack name and digest must be strings")
    pack_names = [pack["name"].lower() for pack in packs]
    if len(pack_names) != len(set(pack_names)):
        raise ValueError("governance pack names must be unique")

    denied_packs = payload["deniedPacks"]
    if not isinstance(denied_packs, list) or not all(isinstance(pack, str) for pack in denied_packs):
        raise ValueError("governance deniedPacks must be a string list")
    if denied_packs != sorted(set(denied_packs)):
        raise ValueError("governance deniedPacks must be sorted and unique")
    if any(pack != pack.lower() or "/" in pack or "\\" in pack for pack in denied_packs):
        raise ValueError("governance deniedPacks entries must be lowercase basenames")
    if denied_packs and custom_node_mode != "blocklist":
        raise ValueError("governance deniedPacks requires blocklist mode")

    disabled_nodes = payload["disabledNodes"]
    if not isinstance(disabled_nodes, list) or not all(isinstance(node_id, str) for node_id in disabled_nodes):
        raise ValueError("governance disabledNodes must be a string list")
    if disabled_nodes != sorted(set(disabled_nodes)):
        raise ValueError("governance disabledNodes must be sorted and unique")

    partner_nodes = payload["disabledPartnerNodes"]
    if not isinstance(partner_nodes, list):
        raise ValueError("governance disabledPartnerNodes must be a list")
    partner_node_ids = []
    for partner_node in partner_nodes:
        if not isinstance(partner_node, dict) or set(partner_node) != {"nodeId", "providerId"}:
            raise ValueError("governance partner-node entries must contain exactly nodeId and providerId")
        if not isinstance(partner_node["nodeId"], str) or not isinstance(partner_node["providerId"], str):
            raise ValueError("governance partner-node ids must be strings")
        if _PROVIDER_ID_PATTERN.fullmatch(partner_node["providerId"]) is None:
            raise ValueError("governance providerId is invalid")
        partner_node_ids.append(partner_node["nodeId"])
    if partner_node_ids != sorted(set(partner_node_ids)):
        raise ValueError("governance partner-node entries must be unique and sorted by nodeId")

    models = payload["models"]
    if not isinstance(models, list) or not all(isinstance(model, str) for model in models):
        raise ValueError("governance models must be a string list")
    if models != sorted(set(models)):
        raise ValueError("governance models must be sorted and unique")
    if any(_MODEL_DIGEST_PATTERN.fullmatch(model) is None for model in models):
        raise ValueError("governance models must contain canonical BLAKE3 digests")

    if packs and not custom_node_active:
        raise ValueError("governance packs requires an active customNode form")
    if disabled_nodes and "nodeId" not in active_forms:
        raise ValueError("governance disabledNodes requires an active nodeId form")
    if "nodeId" in active_forms and not disabled_nodes:
        raise ValueError("governance nodeId form requires non-empty disabledNodes")
    if partner_nodes and "partnerNode" not in active_forms:
        raise ValueError("governance disabledPartnerNodes requires an active partnerNode form")
    if models and "model" not in active_forms:
        raise ValueError("governance models requires an active model form")


def _apply_policy(policy: dict) -> None:
    global _policy, _disabled_nodes
    _policy = policy
    active_forms = policy.get("activeForms", ())
    custom_node_mode = policy.get("customNodeMode") if "customNode" in active_forms else None
    allowed_packs = {pack["name"]: pack["digest"] for pack in policy.get("packs", ())}
    set_custom_node_policy(custom_node_mode, frozenset(policy.get("deniedPacks", ())), allowed_packs)
    partner_provider_map = {entry["nodeId"]: entry["providerId"] for entry in policy.get("disabledPartnerNodes", ())}
    set_partner_provider_map(partner_provider_map)
    _disabled_nodes = frozenset(policy.get("disabledNodes", ())).union(partner_provider_map)


def initialize() -> None:
    if not GOVERNANCE_REQUIRED:
        return

    try:
        from comfy.cli_args import args

        if args.disabled_nodes_config:
            raise RuntimeError("unsigned disabled-node config is not allowed in a governed build")
        if args.extra_model_paths_config:
            raise RuntimeError("unsigned extra-model-paths config is not allowed in a governed build")
        if _EXTRA_MODEL_PATHS_CONFIG_PATH.is_file():
            raise RuntimeError("unsigned extra_model_paths.yaml is not allowed in a governed build")

        policy = verify_and_load(_POLICY_PATH.read_bytes())
        unenforced = sorted(set(policy.get("activeForms", ())).difference(_ENFORCED_FORMS))
        if unenforced:
            raise RuntimeError("policy requires forms this build cannot enforce: " + ", ".join(unenforced))
        _apply_policy(policy)
        if _custom_node_mode == "allowlist" and args.enable_manager:
            raise RuntimeError("ComfyUI-Manager cannot be enabled under a custom-node allowlist")
    except Exception:
        logging.exception("ComfyUI could not apply your organization's policy. Contact your administrator.")
        sys.exit(1)

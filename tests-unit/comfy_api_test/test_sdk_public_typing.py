import ast
from pathlib import Path

from comfy_api.latest import _sdk_public
from comfy_api.generate_api_spec import ARTIFACT, generate_manifest


def test_public_ref_and_domain_operations_are_present_in_the_typing_contract():
    root = Path(_sdk_public.__file__).parent
    implementation = ast.parse((root / "_sdk.py").read_text())
    contract = ast.parse((root / "_sdk_public.pyi").read_text())
    declarations = {
        item.name: item for item in contract.body if isinstance(item, ast.ClassDef)
    }
    failures = []
    for item in implementation.body:
        if not isinstance(item, ast.ClassDef) or item.name not in _sdk_public.__all__:
            continue
        declared = declarations.get(item.name)
        if declared is None:
            failures.append(item.name)
            continue
        methods = {
            member.name: member
            for member in declared.body
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for member in item.body:
            if not isinstance(
                member, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) or member.name.startswith("_"):
                continue
            actual = methods.get(member.name)
            if actual is None:
                failures.append(f"{item.name}.{member.name}")
            elif (
                isinstance(actual, ast.AsyncFunctionDef)
                != isinstance(member, ast.AsyncFunctionDef)
                or [arg.arg for arg in actual.args.args]
                != [arg.arg for arg in member.args.args]
                or [arg.arg for arg in actual.args.kwonlyargs]
                != [arg.arg for arg in member.args.kwonlyargs]
            ):
                failures.append(f"{item.name}.{member.name}: call shape")
    assert failures == []


def test_machine_readable_api_spec_matches_the_public_implementation():
    assert ARTIFACT.read_text() == generate_manifest()


def test_machine_readable_api_includes_nested_io_and_async_storage_call_shapes():
    import json

    specification = json.loads(generate_manifest())["modules"]
    options = specification["io"]["RemoteOptions"]["members"]["__init__"]
    assert "initial_selection" in {
        parameter["name"] for parameter in options["parameters"]
    }
    assert "Input" in specification["io"]["Float"]["members"]
    cas = specification["sdk"]["StorageDomain"]["members"]["compare_and_set_value"]
    assert cas["async"]
    ttl = next(
        parameter
        for parameter in cas["parameters"]
        if parameter["name"] == "ttl_seconds"
    )
    assert ttl["kind"] == "KEYWORD_ONLY" and not ttl["required"]

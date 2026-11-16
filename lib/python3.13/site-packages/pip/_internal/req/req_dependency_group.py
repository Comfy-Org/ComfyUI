import sys
from typing import Any, Dict, Iterable, Iterator, List, Tuple

if sys.version_info >= (3, 11):
    import tomllib
else:
    from pip._vendor import tomli as tomllib

from pip._vendor.dependency_groups import DependencyGroupResolver

from pip._internal.exceptions import InstallationError


def parse_dependency_groups(groups: List[Tuple[str, str]]) -> List[str]:
    """
    Parse dependency groups data as provided via the CLI, in a `[path:]group` syntax.

    Raises InstallationErrors if anything goes wrong.
    """
    resolvers = _build_resolvers(path for (path, _) in groups)
    return list(_resolve_all_groups(resolvers, groups))


def _resolve_all_groups(
    resolvers: Dict[str, DependencyGroupResolver], groups: List[Tuple[str, str]]
) -> Iterator[str]:
    """
    Run all resolution, converting any error from `DependencyGroupResolver` into
    an InstallationError.
    """
    for path, groupname in groups:
        resolver = resolvers[path]
        try:
            yield from (str(req) for req in resolver.resolve(groupname))
        except (ValueError, TypeError, LookupError) as e:
            raise InstallationError(
                f"[dependency-groups] resolution failed for '{groupname}' "
                f"from '{path}': {e}"
            ) from e


def _build_resolvers(paths: Iterable[str]) -> Dict[str, Any]:
    resolvers = {}
    for path in paths:
        if path in resolvers:
            continue

        pyproject = _load_pyproject(path)
        if "dependency-groups" not in pyproject:
            raise InstallationError(
                f"[dependency-groups] table was missing from '{path}'. "
                "Cannot resolve '--group' option."
            )
        raw_dependency_groups = pyproject["dependency-groups"]
        if not isinstance(raw_dependency_groups, dict):
            raise InstallationError(
                f"[dependency-groups] table was malformed in {path}. "
                "Cannot resolve '--group' option."
            )

        resolvers[path] = DependencyGroupResolver(raw_dependency_groups)
    return resolvers


def _load_pyproject(path: str) -> Dict[str, Any]:
    """
    This helper loads a pyproject.toml as TOML.

    It raises an InstallationError if the operation fails.
    """
    try:
        with open(path, "rb") as fp:
            return tomllib.load(fp)
    except FileNotFoundError:
        raise InstallationError(f"{path} not found. Cannot resolve '--group' option.")
    except tomllib.TOMLDecodeError as e:
        raise InstallationError(f"Error parsing {path}: {e}") from e
    except OSError as e:
        raise InstallationError(f"Error reading {path}: {e}") from e

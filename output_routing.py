import datetime
import json
import ntpath
import os
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Union


_PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_VARIABLE_NAMES = frozenset({
    "width", "height", "prefix_dir", "prefix_stem",
})
_ALLOWED_STRFTIME_DIRECTIVES = frozenset({"%", "Y", "y", "m", "d", "H", "M", "S", "j", "U", "W", "V"})


class OutputRoutingError(ValueError):
    pass


@dataclass(frozen=True)
class TemplateLiteral:
    value: str


@dataclass(frozen=True)
class TemplateVariable:
    name: str
    date_format: Optional[str] = None


TemplatePart = Union[TemplateLiteral, TemplateVariable]


@dataclass(frozen=True)
class OutputProfile:
    folder_template: str
    folder_template_parts: tuple[TemplatePart, ...]
    filename_template: Optional[str] = None
    filename_template_parts: Optional[tuple[TemplatePart, ...]] = None


@dataclass(frozen=True)
class OutputRoutingPolicy:
    defaults: Mapping[str, str]
    profiles: Mapping[str, OutputProfile]
    output_directory: Optional[str]

    def get_profile(self, output_type: str) -> OutputProfile:
        return self.profiles[self.defaults[output_type]]


@dataclass(frozen=True)
class OutputRouteContext:
    width: int
    height: int
    prefix_dir: str
    prefix_stem: str


def _legacy_profile() -> OutputProfile:
    return OutputProfile("{prefix_dir}", (TemplateVariable("prefix_dir"),))


def legacy_policy() -> OutputRoutingPolicy:
    profiles = MappingProxyType({"legacy": _legacy_profile()})
    return OutputRoutingPolicy(MappingProxyType({"output": "legacy", "temp": "legacy"}), profiles, None)


def load_policy(path: str) -> OutputRoutingPolicy:
    try:
        with open(path, "r", encoding="utf-8") as policy_file:
            data = json.load(policy_file, object_pairs_hook=_reject_duplicate_keys)
    except OSError as exc:
        raise OutputRoutingError(f"Unable to read output routing policy '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OutputRoutingError(f"Invalid JSON in output routing policy '{path}': {exc}") from exc
    return _parse_policy(data)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    data = {}
    for key, value in pairs:
        if key in data:
            raise OutputRoutingError(f"Duplicate key '{key}' in output routing policy")
        data[key] = value
    return data


def _parse_policy(data: object) -> OutputRoutingPolicy:
    policy = _expect_mapping(data, "Output routing policy")
    _reject_unknown_keys(policy, {"version", "output_directory", "defaults", "profiles"}, "Output routing policy")
    if type(policy.get("version")) is not int or policy["version"] != 1:
        raise OutputRoutingError("Output routing policy version must be 1")

    defaults = {"output": "legacy", "temp": "legacy"}
    defaults.update(_parse_defaults(policy.get("defaults", {})))
    profiles = {"legacy": _legacy_profile()}
    profiles.update(_parse_profiles(policy.get("profiles", {})))
    output_directory = policy.get("output_directory")
    if output_directory is not None:
        _validate_output_directory(output_directory)

    for output_type, profile_name in defaults.items():
        if profile_name not in profiles:
            raise OutputRoutingError(f"Output routing default for '{output_type}' references unknown profile '{profile_name}'")
    return OutputRoutingPolicy(MappingProxyType(defaults), MappingProxyType(profiles), output_directory)


def _validate_output_directory(output_directory: object) -> None:
    if not isinstance(output_directory, str) or not output_directory:
        raise OutputRoutingError("Output routing policy output_directory must be a non-empty path")
    if "\0" in output_directory:
        raise OutputRoutingError("Output routing policy output_directory must not contain a null byte")
    if os.path.isabs(output_directory) or ntpath.isabs(output_directory):
        return
    if ntpath.splitdrive(output_directory)[0]:
        raise OutputRoutingError("Output routing policy output_directory must not be drive-qualified unless it is absolute")
    if any(part == ".." for part in output_directory.replace("\\", "/").split("/")):
        raise OutputRoutingError("Output routing policy relative output_directory must not contain parent-directory segments")


def _parse_defaults(data: object) -> dict[str, str]:
    defaults = _expect_mapping(data, "Output routing defaults")
    _reject_unknown_keys(defaults, {"output", "temp"}, "Output routing defaults")
    parsed = {}
    for output_type, profile_name in defaults.items():
        if not isinstance(profile_name, str) or not _PROFILE_NAME_PATTERN.fullmatch(profile_name):
            raise OutputRoutingError(f"Output routing default for '{output_type}' must be a valid profile name")
        parsed[output_type] = profile_name
    return parsed


def _parse_profiles(data: object) -> dict[str, OutputProfile]:
    raw_profiles = _expect_mapping(data, "Output routing profiles")
    profiles = {}
    for profile_name, profile_data in raw_profiles.items():
        if profile_name == "legacy":
            raise OutputRoutingError("The built-in 'legacy' output routing profile cannot be redefined")
        if not isinstance(profile_name, str) or not _PROFILE_NAME_PATTERN.fullmatch(profile_name):
            raise OutputRoutingError(f"Invalid output routing profile name '{profile_name}'")
        profile = _expect_mapping(profile_data, f"Output routing profile '{profile_name}'")
        _reject_unknown_keys(profile, {"folder_template", "filename_template"}, f"Output routing profile '{profile_name}'")
        folder_template = profile.get("folder_template")
        if not isinstance(folder_template, str):
            raise OutputRoutingError(f"Output routing profile '{profile_name}' requires a string folder_template")
        folder_template_parts = _parse_template(folder_template)
        _validate_template_path(folder_template_parts)

        filename_template = profile.get("filename_template")
        if filename_template is not None and not isinstance(filename_template, str):
            raise OutputRoutingError(f"Output routing profile '{profile_name}' filename_template must be a string")
        filename_template_parts = None
        if filename_template is not None:
            filename_template_parts = _parse_template(filename_template)
            _validate_filename_stem_template(filename_template_parts)

        profiles[profile_name] = OutputProfile(
            folder_template,
            folder_template_parts,
            filename_template,
            filename_template_parts,
        )
    return profiles


def _expect_mapping(data: object, description: str) -> Mapping[str, object]:
    if not isinstance(data, dict):
        raise OutputRoutingError(f"{description} must be an object")
    return data


def _reject_unknown_keys(data: Mapping[str, object], allowed_keys: set[str], description: str) -> None:
    unknown_keys = set(data).difference(allowed_keys)
    if unknown_keys:
        raise OutputRoutingError(f"{description} contains unsupported key(s): {', '.join(sorted(unknown_keys))}")


def _parse_template(template: str) -> tuple[TemplatePart, ...]:
    parts: list[TemplatePart] = []
    position = 0
    while position < len(template):
        start = template.find("{", position)
        if start == -1:
            _append_literal(parts, template[position:])
            break
        _append_literal(parts, template[position:start])
        end = template.find("}", start + 1)
        if end == -1:
            raise OutputRoutingError("Output routing template has an unclosed placeholder")
        parts.append(_parse_template_field(template[start + 1:end]))
        position = end + 1
    return tuple(parts)


def _append_literal(parts: list[TemplatePart], literal: str) -> None:
    if "}" in literal:
        raise OutputRoutingError("Output routing template has an unmatched closing brace")
    if literal:
        parts.append(TemplateLiteral(literal))


def _parse_template_field(field: str) -> TemplateVariable:
    if field.startswith("date:"):
        date_format = field.removeprefix("date:")
        if not date_format:
            raise OutputRoutingError("Output routing date placeholder requires a strftime format")
        _validate_strftime_format(date_format)
        return TemplateVariable("date", date_format)
    if field not in _VARIABLE_NAMES:
        raise OutputRoutingError(f"Unsupported output routing template placeholder '{{{field}}}'")
    return TemplateVariable(field)


def _validate_template_path(parts: tuple[TemplatePart, ...]) -> None:
    _validate_relative_folder("".join(part.value if isinstance(part, TemplateLiteral) else "value" for part in parts))


def _validate_strftime_format(date_format: str) -> None:
    position = 0
    while position < len(date_format):
        if date_format[position] != "%":
            position += 1
            continue
        if position + 1 == len(date_format) or date_format[position + 1] not in _ALLOWED_STRFTIME_DIRECTIVES:
            raise OutputRoutingError(f"Unsupported strftime directive in output routing template: '{date_format}'")
        position += 2


def resolve_route(policy: OutputRoutingPolicy, output_type: str, context: OutputRouteContext) -> tuple[str, str]:
    profile = policy.get_profile(output_type)
    saved_at = datetime.datetime.now()
    subfolder = _validate_relative_folder(_render_template(profile.folder_template_parts, context, saved_at))
    if profile.filename_template_parts is None:
        return subfolder, context.prefix_stem
    filename_stem = _validate_filename_stem(_render_template(profile.filename_template_parts, context, saved_at))
    return subfolder, filename_stem


def _render_template(parts: tuple[TemplatePart, ...], context: OutputRouteContext, saved_at: datetime.datetime) -> str:
    values = {
        "width": context.width,
        "height": context.height,
        "prefix_dir": context.prefix_dir,
        "prefix_stem": context.prefix_stem,
    }
    rendered = []
    for part in parts:
        if isinstance(part, TemplateLiteral):
            rendered.append(part.value)
        elif part.name == "date":
            rendered.append(saved_at.strftime(part.date_format))
        else:
            value = values[part.name]
            if value is None:
                raise OutputRoutingError(f"Output routing template requires unavailable value '{part.name}'")
            rendered.append(str(value))
    return "".join(rendered)


def _validate_relative_folder(folder: str) -> str:
    if "\0" in folder:
        raise OutputRoutingError("Output routing template resolved to a path containing a null byte")
    if os.path.isabs(folder) or ntpath.isabs(folder) or ntpath.splitdrive(folder)[0]:
        raise OutputRoutingError("Output routing template must resolve to a relative path")
    normalized_separators = folder.replace("\\", "/")
    if any(part == ".." for part in normalized_separators.split("/")):
        raise OutputRoutingError("Output routing template must not contain parent-directory segments")
    normalized = os.path.normpath(normalized_separators)
    return "" if normalized == "." else normalized


def _validate_filename_stem_template(parts: tuple[TemplatePart, ...]) -> None:
    _validate_filename_stem("".join(part.value if isinstance(part, TemplateLiteral) else "value" for part in parts))


def _validate_filename_stem(filename_stem: str) -> str:
    if not filename_stem or filename_stem in (".", ".."):
        raise OutputRoutingError("Output routing filename_template must resolve to a non-empty filename stem")
    if "\0" in filename_stem:
        raise OutputRoutingError("Output routing filename_template resolved to a filename containing a null byte")
    if os.path.isabs(filename_stem) or ntpath.isabs(filename_stem) or ntpath.splitdrive(filename_stem)[0]:
        raise OutputRoutingError("Output routing filename_template must resolve to a filename stem, not a path")
    if "/" in filename_stem or "\\" in filename_stem:
        raise OutputRoutingError("Output routing filename_template must not contain path separators")
    if "." in filename_stem:
        raise OutputRoutingError("Output routing filename_template must not contain a file extension")
    if any(character in filename_stem for character in '<>:"|?*'):
        raise OutputRoutingError("Output routing filename_template contains a filename character invalid on Windows")
    if filename_stem.rstrip(". ") != filename_stem:
        raise OutputRoutingError("Output routing filename_template must not end with a period or space")
    return filename_stem

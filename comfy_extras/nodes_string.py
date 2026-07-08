import re
import json
import string
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class StringFormat(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow = io.Autogrow.TemplateNames(
            input=io.AnyType.Input("value"),
            names=list(string.ascii_lowercase),
            min=0,
        )
        return io.Schema(
            node_id="StringFormat",
            display_name="Format Text",
            category="text",
            search_aliases=["string", "format"],
            description="Same as Python's string format method. Supports all of Python's format options and features.",
            inputs=[
                io.Autogrow.Input("values", template=autogrow),
                io.String.Input("f_string", default="{a}", multiline=True),
            ],
            outputs=[
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(
        cls, values: io.Autogrow.Type, f_string: str
    ) -> io.NodeOutput:
        return io.NodeOutput(f_string.format(**values))


class StringConcatenate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringConcatenate",
            search_aliases=["concatenate", "text concat", "join text", "merge text", "combine strings", "string concat", "append text", "combine text"],
            display_name="Concatenate Text",
            category="text",
            inputs=[
                io.String.Input("string_a", multiline=True),
                io.String.Input("string_b", multiline=True),
                io.String.Input("delimiter", multiline=False, default=""),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string_a, string_b, delimiter):
        return io.NodeOutput(delimiter.join((string_a, string_b)))


class StringSubstring(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringSubstring",
            search_aliases=["substring", "extract text", "text portion"],
            display_name="Substring",
            category="text",
            inputs=[
                io.String.Input("string", multiline=True),
                io.Int.Input("start"),
                io.Int.Input("end"),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, start, end):
        return io.NodeOutput(string[start:end])


class StringLength(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringLength",
            search_aliases=["character count", "text size", "string length"],
            display_name="Text Length",
            category="text",
            inputs=[
                io.String.Input("string", multiline=True),
            ],
            outputs=[
                io.Int.Output(display_name="length"),
            ]
        )

    @classmethod
    def execute(cls, string):
        return io.NodeOutput(len(string))


class CaseConverter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CaseConverter",
            search_aliases=["case converter", "text case", "uppercase", "lowercase", "capitalize"],
            display_name="Convert Text Case",
            category="text",
            inputs=[
                io.String.Input("string", multiline=True),
                io.Combo.Input("mode", options=["UPPERCASE", "lowercase", "Capitalize", "Title Case"]),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, mode):
        if mode == "UPPERCASE":
            result = string.upper()
        elif mode == "lowercase":
            result = string.lower()
        elif mode == "Capitalize":
            result = string.capitalize()
        elif mode == "Title Case":
            result = string.title()
        else:
            result = string

        return io.NodeOutput(result)


class StringTrim(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringTrim",
            search_aliases=["trim", "clean whitespace", "remove whitespace", "remove spaces","strip"],
            display_name="Trim Text",
            category="text",
            inputs=[
                io.String.Input("string", multiline=True),
                io.Combo.Input("mode", options=["Both", "Left", "Right"]),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, mode):
        if mode == "Both":
            result = string.strip()
        elif mode == "Left":
            result = string.lstrip()
        elif mode == "Right":
            result = string.rstrip()
        else:
            result = string

        return io.NodeOutput(result)


class StringReplace(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringReplace",
            search_aliases=["replace", "find and replace", "substitute", "swap text"],
            display_name="Replace Text",
            category="text",
            inputs=[
                io.String.Input("string", multiline=True),
                io.String.Input("find", multiline=True),
                io.String.Input("replace", multiline=True),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, find, replace):
        return io.NodeOutput(string.replace(find, replace))


class StringContains(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringContains",
            search_aliases=["contains", "text includes", "string includes"],
            display_name="Contains Text",
            category="text",
            inputs=[
                io.String.Input("string", multiline=True),
                io.String.Input("substring", multiline=True),
                io.Boolean.Input("case_sensitive", default=True, advanced=True),
            ],
            outputs=[
                io.Boolean.Output(display_name="contains"),
            ]
        )

    @classmethod
    def execute(cls, string, substring, case_sensitive):
        if case_sensitive:
            contains = substring in string
        else:
            contains = substring.lower() in string.lower()

        return io.NodeOutput(contains)


class StringCompare(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringCompare",
            search_aliases=["compare", "text match", "string equals", "starts with", "ends with"],
            display_name="Compare Text",
            category="text",
            inputs=[
                io.String.Input("string_a", multiline=True),
                io.String.Input("string_b", multiline=True),
                io.Combo.Input("mode", options=["Starts With", "Ends With", "Equal"]),
                io.Boolean.Input("case_sensitive", default=True, advanced=True),
            ],
            outputs=[
                io.Boolean.Output(),
            ]
        )

    @classmethod
    def execute(cls, string_a, string_b, mode, case_sensitive):
        if case_sensitive:
            a = string_a
            b = string_b
        else:
            a = string_a.lower()
            b = string_b.lower()

        if mode == "Equal":
            return io.NodeOutput(a == b)
        elif mode == "Starts With":
            return io.NodeOutput(a.startswith(b))
        elif mode == "Ends With":
            return io.NodeOutput(a.endswith(b))


class RegexMatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RegexMatch",
            search_aliases=["regex match", "regex", "pattern match", "text contains", "string match"],
            display_name="Match Text",
            category="text",
            inputs=[
                io.String.Input("string", multiline=True),
                io.String.Input("regex_pattern", multiline=True),
                io.Boolean.Input("case_insensitive", default=True, advanced=True),
                io.Boolean.Input("multiline", default=False, advanced=True),
                io.Boolean.Input("dotall", default=False, advanced=True),
            ],
            outputs=[
                io.Boolean.Output(display_name="matches"),
            ]
        )

    @classmethod
    def execute(cls, string, regex_pattern, case_insensitive, multiline, dotall):
        flags = 0

        if case_insensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL

        try:
            match = re.search(regex_pattern, string, flags)
            result = match is not None

        except re.error:
            result = False

        return io.NodeOutput(result)


class RegexExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RegexExtract",
            search_aliases=["regex extract", "regex", "pattern extract", "text parser", "parse text"],
            display_name="Extract Text",
            category="text",
            inputs=[
                io.String.Input("string", multiline=True),
                io.String.Input("regex_pattern", multiline=True),
                io.Combo.Input("mode", options=["First Match", "All Matches", "First Group", "All Groups"]),
                io.Boolean.Input("case_insensitive", default=True, advanced=True),
                io.Boolean.Input("multiline", default=False, advanced=True),
                io.Boolean.Input("dotall", default=False, advanced=True),
                io.Int.Input("group_index", default=1, min=0, max=100, advanced=True),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, regex_pattern, mode, case_insensitive, multiline, dotall, group_index):
        join_delimiter = "\n"

        flags = 0
        if case_insensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL

        try:
            if mode == "First Match":
                match = re.search(regex_pattern, string, flags)
                if match:
                    result = match.group(0)
                else:
                    result = ""

            elif mode == "All Matches":
                matches = re.findall(regex_pattern, string, flags)
                if matches:
                    if isinstance(matches[0], tuple):
                        result = join_delimiter.join([m[0] for m in matches])
                    else:
                        result = join_delimiter.join(matches)
                else:
                    result = ""

            elif mode == "First Group":
                match = re.search(regex_pattern, string, flags)
                if match and len(match.groups()) >= group_index:
                    result = match.group(group_index)
                else:
                    result = ""

            elif mode == "All Groups":
                matches = re.finditer(regex_pattern, string, flags)
                results = []
                for match in matches:
                    if match.groups() and len(match.groups()) >= group_index:
                        results.append(match.group(group_index))
                result = join_delimiter.join(results)
            else:
                result = ""

        except re.error:
            result = ""

        return io.NodeOutput(result)


class RegexReplace(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RegexReplace",
            search_aliases=["regex replace", "regex", "pattern replace", "substitution"],
            display_name="Replace Text (Regex)",
            category="text",
            description="Find and replace text using regex patterns.",
            inputs=[
                io.String.Input("string", multiline=True),
                io.String.Input("regex_pattern", multiline=True),
                io.String.Input("replace", multiline=True),
                io.Boolean.Input("case_insensitive", default=True, optional=True, advanced=True),
                io.Boolean.Input("multiline", default=False, optional=True, advanced=True),
                io.Boolean.Input("dotall", default=False, optional=True, advanced=True, tooltip="When enabled, the dot (.) character will match any character including newline characters. When disabled, dots won't match newlines."),
                io.Int.Input("count", default=0, min=0, max=100, optional=True, advanced=True, tooltip="Maximum number of replacements to make. Set to 0 to replace all occurrences (default). Set to 1 to replace only the first match, 2 for the first two matches, etc."),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, string, regex_pattern, replace, case_insensitive=True, multiline=False, dotall=False, count=0):
        flags = 0

        if case_insensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL
        result = re.sub(regex_pattern, replace, string, count=count, flags=flags)
        return io.NodeOutput(result)


class JsonExtractString(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JsonExtractString",
            display_name="Extract Text from JSON",
            category="text",
            search_aliases=["json", "extract json", "parse json", "json value", "read json"],
            inputs=[
                io.String.Input("json_string", multiline=True),
                io.String.Input("key", multiline=False),
            ],
            outputs=[
                io.String.Output(),
            ]
        )

    @classmethod
    def execute(cls, json_string, key):
        try:
            data = json.loads(json_string)
            if isinstance(data, dict) and key in data:
                value = data[key]
                if value is None:
                    return io.NodeOutput("")

                return io.NodeOutput(str(value))

            return io.NodeOutput("")

        except (json.JSONDecodeError, TypeError):
            return io.NodeOutput("")


def _dump_json(value, indent):
    return json.dumps(value, ensure_ascii=False, indent=indent or None)


class ConvertDictionaryToString(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConvertDictionaryToString",
            display_name="Convert Dictionary to String",
            category="text",
            search_aliases=["json", "dict to json", "stringify", "serialize", "dict to string"],
            inputs=[
                io.Dict.Input("dictionary"),
                io.Int.Input("indent", default=2, min=0, max=8,
                             tooltip="Spaces per indent level. 0 produces compact single-line string."),
            ],
            outputs=[
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, dictionary, indent=2):
        return io.NodeOutput(_dump_json(dictionary, indent))


class ConvertArrayToString(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConvertArrayToString",
            display_name="Convert Array to String",
            category="text",
            search_aliases=["json", "list to json", "stringify", "serialize", "list to string", "array to json"],
            inputs=[
                io.Array.Input("array"),
                io.Int.Input("indent", default=2, min=0, max=8,
                             tooltip="Spaces per indent level. 0 produces compact single-line string."),
            ],
            outputs=[
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, array, indent=2):
        return io.NodeOutput(_dump_json(array, indent))


class StringExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            StringFormat,
            StringConcatenate,
            StringSubstring,
            StringLength,
            CaseConverter,
            StringTrim,
            StringReplace,
            StringContains,
            StringCompare,
            RegexMatch,
            RegexExtract,
            RegexReplace,
            JsonExtractString,
            ConvertDictionaryToString,
            ConvertArrayToString,
        ]

async def comfy_entrypoint() -> StringExtension:
    return StringExtension()

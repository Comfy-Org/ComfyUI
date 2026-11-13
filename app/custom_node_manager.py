import os
import folder_paths
import glob
from aiohttp import web
import json
import logging
from functools import lru_cache

from utils.json_util import merge_json_recursive


# Extra locale files to load into main.json
EXTRA_LOCALE_FILES = [
    "nodeDefs.json",
    "commands.json",
    "settings.json",
]

# Folder names, in priority order, that a custom node's example workflows may
# live under. "example_workflows" is the preferred/current convention; the
# rest are legacy aliases still supported for backward compatibility.
EXAMPLE_WORKFLOW_FOLDER_NAMES = [
    "example_workflows",
    "example",
    "examples",
    "workflow",
    "workflows",
]


def safe_load_json_file(file_path: str) -> dict:
    if not os.path.exists(file_path):
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Error loading {file_path}")
        return {}


def collect_workflow_templates() -> dict[str, dict[str, str]]:
    """Scans every installed custom node for example workflow JSON files
    across all recognized folder-name conventions.

    Returns a mapping of custom-node directory name -> {visible template
    name -> absolute file path}, preserving discovery order.

    Two things are handled that a naive glob-and-concatenate would get wrong:

    - A custom node that exposes the same physical directory under more than
      one recognized alias (most commonly a symlink such as
      `example_workflows -> examples`) has that directory scanned only once,
      so its templates aren't counted twice.
    - A custom node that has *genuinely different* files sharing the same
      filename across two different alias folders (e.g. `examples/foo.json`
      and `workflow/foo.json`, both real, unrelated files) keeps both, with
      the later one disambiguated by appending the folder it came from, so
      neither is silently dropped or made unreachable.
    """
    templates_by_module: dict[str, dict[str, str]] = {}
    seen_dirs_by_module: dict[str, set[str]] = {}

    for root in folder_paths.get_folder_paths("custom_nodes"):
        for folder_name in EXAMPLE_WORKFLOW_FOLDER_NAMES:
            pattern = os.path.join(root, "*", folder_name)
            for workflows_dir in sorted(glob.glob(pattern)):
                if not os.path.isdir(workflows_dir):
                    continue

                custom_nodes_name = os.path.basename(os.path.dirname(workflows_dir))
                real_dir = os.path.realpath(workflows_dir)

                seen_dirs = seen_dirs_by_module.setdefault(custom_nodes_name, set())
                if real_dir in seen_dirs:
                    # Another recognized alias already resolved to this exact
                    # physical directory (e.g. a symlink) - skip it so its
                    # files aren't counted a second time.
                    continue
                seen_dirs.add(real_dir)

                templates = templates_by_module.setdefault(custom_nodes_name, {})
                for file in sorted(glob.glob(os.path.join(workflows_dir, "*.json"))):
                    workflow_name = os.path.splitext(os.path.basename(file))[0]
                    name = workflow_name
                    suffix = 2
                    while name in templates:
                        name = (
                            f"{workflow_name}-{folder_name}"
                            if suffix == 2
                            else f"{workflow_name}-{folder_name}-{suffix}"
                        )
                        suffix += 1
                    templates[name] = file

    return templates_by_module


class CustomNodeManager:
    @lru_cache(maxsize=1)
    def build_translations(self):
        """Load all custom nodes translations during initialization. Translations are
        expected to be loaded from `locales/` folder.

        The folder structure is expected to be the following:
        - custom_nodes/
            - custom_node_1/
                - locales/
                    - en/
                        - main.json
                        - commands.json
                        - settings.json

        returned translations are expected to be in the following format:
        {
            "en": {
                "nodeDefs": {...},
                "commands": {...},
                "settings": {...},
                ...{other main.json keys}
            }
        }
        """

        translations = {}

        for folder in folder_paths.get_folder_paths("custom_nodes"):
            # Sort glob results for deterministic ordering
            for custom_node_dir in sorted(glob.glob(os.path.join(folder, "*/"))):
                locales_dir = os.path.join(custom_node_dir, "locales")
                if not os.path.exists(locales_dir):
                    continue

                for lang_dir in glob.glob(os.path.join(locales_dir, "*/")):
                    lang_code = os.path.basename(os.path.dirname(lang_dir))

                    if lang_code not in translations:
                        translations[lang_code] = {}

                    # Load main.json
                    main_file = os.path.join(lang_dir, "main.json")
                    node_translations = safe_load_json_file(main_file)

                    # Load extra locale files
                    for extra_file in EXTRA_LOCALE_FILES:
                        extra_file_path = os.path.join(lang_dir, extra_file)
                        key = extra_file.split(".")[0]
                        json_data = safe_load_json_file(extra_file_path)
                        if json_data:
                            node_translations[key] = json_data

                    if node_translations:
                        translations[lang_code] = merge_json_recursive(
                            translations[lang_code], node_translations
                        )

        return translations

    def add_routes(self, routes, webapp, loadedModules):

        @routes.get("/workflow_templates")
        async def get_workflow_templates(request):
            """Returns a web response that contains the map of custom_nodes names and their associated workflow templates. The ones without templates are omitted."""
            templates_by_module = collect_workflow_templates()
            return web.json_response(
                {
                    module: list(templates.keys())
                    for module, templates in templates_by_module.items()
                }
            )

        @routes.get("/workflow_templates/{module_name}/{filename}")
        async def get_workflow_template_file(request):
            """Serves a single example workflow JSON file for a custom node.

            Looked up by the same (module, visible name) pairing used by
            `get_workflow_templates` above, rather than a direct static file
            mount, so that every recognized alias folder for a custom node is
            actually reachable - not just whichever one happened to be
            registered first - and so a collision between two distinct files
            sharing a filename can never be served ambiguously.
            """
            module_name = request.match_info["module_name"]
            filename = request.match_info["filename"]
            if not filename.endswith(".json"):
                raise web.HTTPNotFound()

            name = filename[: -len(".json")]
            templates = collect_workflow_templates().get(module_name)
            if not templates or name not in templates:
                raise web.HTTPNotFound()

            return web.FileResponse(templates[name])

        @routes.get("/i18n")
        async def get_i18n(request):
            """Returns translations from all custom nodes' locales folders."""
            return web.json_response(self.build_translations())

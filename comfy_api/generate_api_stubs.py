#!/usr/bin/env python3
"""
Script to generate .pyi stub files for the synchronous API wrappers.
This allows generating stubs without running the full ComfyUI application.
"""

import os
import sys
import logging
import importlib
import importlib.util

# Add ComfyUI to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfy_api.internal.async_to_sync import AsyncToSyncConverter
from comfy_api.version_list import supported_versions


def generate_stubs_for_module(module_name: str) -> None:
    """Generate stub files for a specific module that exports ComfyAPI and ComfyAPISync."""
    try:
        # Import the module
        module = importlib.import_module(module_name)

        # A version with its own generator writes its own stubs
        if importlib.util.find_spec(f"{module_name}._generate") is not None:
            importlib.import_module(f"{module_name}._generate").main()
            logging.info(f"Generated stub file for {module_name}")
            return

        # Check if module has ComfyAPISync (the sync wrapper)
        if hasattr(module, "ComfyAPISync"):
            # Module already has a sync class
            api_class = getattr(module, "ComfyAPI", None)
            sync_class = getattr(module, "ComfyAPISync")

            if api_class:
                # Generate the stub file
                AsyncToSyncConverter.generate_stub_file(api_class, sync_class)
                logging.info(f"Generated stub file for {module_name}")
            else:
                logging.warning(
                    f"Module {module_name} has ComfyAPISync but no ComfyAPI"
                )

        elif hasattr(module, "ComfyAPI"):
            # Module only has async API, need to create sync wrapper first
            from comfy_api.internal.async_to_sync import create_sync_class

            api_class = getattr(module, "ComfyAPI")
            sync_class = create_sync_class(api_class)

            # Generate the stub file
            AsyncToSyncConverter.generate_stub_file(api_class, sync_class)
            logging.info(f"Generated stub file for {module_name}")
        else:
            logging.warning(
                f"Module {module_name} does not export ComfyAPI or ComfyAPISync"
            )

    except Exception as e:
        logging.error(f"Failed to generate stub for {module_name}: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main function to generate all API stub files."""
    logging.basicConfig(level=logging.INFO)

    logging.info("Starting stub generation...")

    # Dynamically get module names from supported_versions
    api_modules = []
    for api_class in supported_versions:
        # Extract module name from the class
        module_name = api_class.__module__
        if module_name not in api_modules:
            api_modules.append(module_name)

    logging.info(f"Found {len(api_modules)} API modules: {api_modules}")

    # Generate stubs for each module
    for module_name in api_modules:
        generate_stubs_for_module(module_name)

    logging.info("Stub generation complete!")


if __name__ == "__main__":
    main()

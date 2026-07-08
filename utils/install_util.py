from pathlib import Path
import sys
import logging
import re

# The path to the requirements.txt file
requirements_path = Path(__file__).parents[1] / "requirements.txt"


def get_missing_requirements_message():
    """The warning message to display when a package is missing."""

    extra = ""
    if sys.flags.no_user_site:
        extra = "-s "
    return f"""
Please install the updated requirements.txt file by running:
{sys.executable} {extra}-m pip install -r {requirements_path}
If you are on the portable package you can run: update\\update_comfyui.bat to solve this problem.
""".strip()


def is_valid_version(version: str) -> bool:
    """Validate if a string is a valid semantic version (X.Y.Z format)."""
    pattern = r"^(\d+)\.(\d+)\.(\d+)$"
    return bool(re.match(pattern, version))


PACKAGE_VERSIONS = {}
def get_required_packages_versions():
    if len(PACKAGE_VERSIONS) > 0:
        return PACKAGE_VERSIONS.copy()
    out = PACKAGE_VERSIONS
    try:
        with open(requirements_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().replace(">=", "==")
                s = line.split("==")
                if len(s) == 2:
                    version_str = s[-1]
                    if not is_valid_version(version_str):
                        logging.debug(f"Invalid version format for {s[0]} in requirements.txt: {version_str}")
                        continue
                    out[s[0]] = version_str
        return out.copy()
    except FileNotFoundError:
        logging.error("requirements.txt not found.")
        return None
    except Exception as e:
        logging.error(f"Error reading requirements.txt: {e}")
        return None

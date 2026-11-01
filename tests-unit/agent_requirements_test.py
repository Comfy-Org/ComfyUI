"""agent_requirements.txt is read by ComfyUI Desktop as well as by pip.

Desktop's agent version override (Comfy-Org/Comfy-Desktop,
src/main/lib/agentRequirementsOverride.ts) rewrites the agent line on each
launch and refuses the whole override, frontend pin included, for any line
that is not a bare name or `name==version`. A bare `comfy-agent` is what lets
it apply any rollout version, and it never downgrades an installed agent.
"""

import os
import re

REQUIREMENTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_requirements.txt")
DESKTOP_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:==\s*([^\s;#]+))?\s*(?:#.*)?$")


def _lines():
    with open(REQUIREMENTS, encoding="utf-8") as f:
        return [line for line in f.read().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_every_line_is_one_desktop_can_override():
    assert _lines()
    for line in _lines():
        assert DESKTOP_LINE.match(line), f"Desktop refuses its agent override for {line!r}"


def test_the_agent_is_never_pinned():
    # An exact pin downgrades a newer agent on `pip install -r`, and Desktop
    # refuses a rollout version that is not newer than it.
    agent = [line for line in _lines() if DESKTOP_LINE.match(line).group(1) == "comfy-agent"]
    assert agent == ["comfy-agent"]

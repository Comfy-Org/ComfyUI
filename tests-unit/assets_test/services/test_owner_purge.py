import subprocess
from pathlib import Path


def test_no_owner_id_in_asset_modules():
    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["grep", "-rn", "--exclude=*.pyc", "owner_id", "app/assets/"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""

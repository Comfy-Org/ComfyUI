import pytest


@pytest.fixture
def comfy_environment(monkeypatch):
    from comfy.cli_args import args

    monkeypatch.setattr(args, "cpu", True)

import runpy
import shutil
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("platform", "bash", "skip"),
    [
        ("win32", "C:/Windows/System32/bash.exe", True),
        ("win32", None, True),
        ("linux", "/usr/bin/bash", False),
        ("linux", None, True),
        ("darwin", "/bin/bash", False),
    ],
)
def test_deployment_platform_guard(monkeypatch, platform, bash, skip):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(shutil, "which", lambda executable: bash)

    deployment_tests = runpy.run_path(str(Path(__file__).with_name("test_docs_deployment.py")))

    assert deployment_tests["pytestmark"].args[0] is skip

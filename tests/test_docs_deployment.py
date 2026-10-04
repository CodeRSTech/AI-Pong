import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/docs.yml"
pytestmark = pytest.mark.skipif(
    not shutil.which("bash"),
    reason="Deployment runs on Linux and requires bash",
)


@pytest.fixture
def assembly_script():
    step = WORKFLOW.read_text().split("      - name: Assemble Pages artifact\n", 1)[1]
    run = step.split("        run: |\n", 1)[1].split("\n      - uses:", 1)[0]
    return textwrap.dedent(run)


def assemble_site(root, script):
    return subprocess.run(["bash", "-e", "-c", script], cwd=root, capture_output=True, text=True)


def test_assembly_publishes_homepage_and_docs(tmp_path, assembly_script):
    (tmp_path / "index.html").write_text("AI-Pong homepage")
    docs = tmp_path / "site/docs"
    docs.mkdir(parents=True)
    (docs / "index.html").write_text("AI-Pong docs")
    (docs / "guide.html").write_text("guide")

    result = assemble_site(tmp_path, assembly_script)

    assert result.returncode == 0, result.stderr
    site = tmp_path / "site"
    assert (site / "index.html").read_text() == "AI-Pong homepage"
    assert (site / "docs/index.html").read_text() == "AI-Pong docs"
    assert (site / "docs/guide.html").read_text() == "guide"


@pytest.mark.parametrize("missing", ["index.html", "site/docs/index.html"])
def test_assembly_rejects_missing_homepage_or_docs(tmp_path, assembly_script, missing):
    docs = tmp_path / "site/docs"
    docs.mkdir(parents=True)
    (tmp_path / "index.html").write_text("homepage")
    (docs / "index.html").write_text("docs")
    (tmp_path / missing).unlink()

    result = assemble_site(tmp_path, assembly_script)

    assert result.returncode != 0
    assert not (tmp_path / "site/index.html").exists()

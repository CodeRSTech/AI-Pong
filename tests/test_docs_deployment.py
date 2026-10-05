import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/docs.yml"
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not shutil.which("bash"),
    reason="Deployment runs on Linux and requires bash",
)


@pytest.fixture
def assembly_script():
    step = WORKFLOW.read_text().split("      - name: Assemble Pages artifact\n", 1)[1]
    run = step.split("        run: |\n", 1)[1].split("\n      - uses:", 1)[0]
    return textwrap.dedent(run)


def assemble_site(root, script):
    return subprocess.run(["bash", "-e", "-c", script], cwd=root, capture_output=True, text=True)


def create_assets(root):
    assets = root / "assets"
    assets.mkdir()
    for name in ("demo.gif", "model.json", "pong-core.js", "pong-demo.js"):
        (assets / name).write_text(f"asset {name}")


def test_assembly_publishes_homepage_and_docs(tmp_path, assembly_script):
    create_assets(tmp_path)
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
    for name in ("demo.gif", "model.json", "pong-core.js", "pong-demo.js"):
        assert (site / "assets" / name).read_text() == f"asset {name}"


@pytest.mark.parametrize("missing", ["index.html", "site/docs/index.html"])
def test_assembly_rejects_missing_homepage_or_docs(tmp_path, assembly_script, missing):
    create_assets(tmp_path)
    docs = tmp_path / "site/docs"
    docs.mkdir(parents=True)
    (tmp_path / "index.html").write_text("homepage")
    (docs / "index.html").write_text("docs")
    (tmp_path / missing).unlink()

    result = assemble_site(tmp_path, assembly_script)

    assert result.returncode != 0
    assert not (tmp_path / "site/index.html").exists()


@pytest.mark.parametrize("missing", ["demo.gif", "model.json", "pong-core.js", "pong-demo.js"])
def test_assembly_rejects_missing_demo_assets(tmp_path, assembly_script, missing):
    create_assets(tmp_path)
    (tmp_path / "assets" / missing).unlink()
    (tmp_path / "index.html").write_text("homepage")
    docs = tmp_path / "site/docs"
    docs.mkdir(parents=True)
    (docs / "index.html").write_text("docs")

    result = assemble_site(tmp_path, assembly_script)

    assert result.returncode != 0
    assert not (tmp_path / "site/index.html").exists()

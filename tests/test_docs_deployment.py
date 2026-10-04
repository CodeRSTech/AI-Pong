import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/docs.yml"
pytestmark = pytest.mark.skipif(
    not shutil.which("bash") or not shutil.which("rsync"),
    reason="Deployment runs on Linux and requires bash and rsync",
)


@pytest.fixture
def sync_script():
    # Exercise the actual workflow step without credentials or a remote push.
    step = WORKFLOW.read_text().split("      - name: Sync documentation\n", 1)[1]
    run = step.split("        run: |\n", 1)[1].split("\n      - name:", 1)[0]
    return textwrap.dedent(run)


def sync_docs(root, script):
    return subprocess.run(["bash", "-e", "-c", script], cwd=root, capture_output=True, text=True)


def test_sync_preserves_landing_page_and_portfolio(tmp_path, sync_script):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("new docs")
    (site / "css").mkdir()
    (site / "css/theme.css").write_text("body {}")
    portfolio = tmp_path / "pages-repository"
    docs = portfolio / "projects/AI-Pong/docs"
    docs.mkdir(parents=True)
    (docs / "obsolete.html").write_text("old docs")
    (docs.parent / "index.html").write_text("landing page")
    (portfolio / "_config.yml").write_text("theme: jekyll-theme-hacker")
    (portfolio / "index.html").write_text("portfolio")

    result = sync_docs(tmp_path, sync_script)

    assert result.returncode == 0, result.stderr
    assert (docs / "index.html").read_text() == "new docs"
    assert (docs / "css/theme.css").read_text() == "body {}"
    assert not (docs / "obsolete.html").exists()
    assert (docs.parent / "index.html").read_text() == "landing page"
    assert (portfolio / "index.html").read_text() == "portfolio"
    assert (portfolio / "_config.yml").read_text() == "theme: jekyll-theme-hacker"
    assert not (portfolio / ".nojekyll").exists()
    assert sync_docs(tmp_path, sync_script).returncode == 0


def test_sync_creates_docs_directory(tmp_path, sync_script):
    (tmp_path / "site").mkdir()
    (tmp_path / "site/index.html").write_text("docs")
    (tmp_path / "pages-repository").mkdir()

    result = sync_docs(tmp_path, sync_script)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "pages-repository/projects/AI-Pong/docs/index.html").read_text() == "docs"


@pytest.mark.parametrize("index_content", [None, ""])
def test_sync_rejects_missing_or_empty_build(tmp_path, sync_script, index_content):
    (tmp_path / "site").mkdir()
    if index_content is not None:
        (tmp_path / "site/index.html").write_text(index_content)
    docs = tmp_path / "pages-repository/projects/AI-Pong/docs"
    docs.mkdir(parents=True)
    (docs / "index.html").write_text("existing docs")

    assert sync_docs(tmp_path, sync_script).returncode != 0
    assert (docs / "index.html").read_text() == "existing docs"


@pytest.mark.parametrize("target", ["projects", "projects/AI-Pong", "projects/AI-Pong/docs"])
def test_sync_rejects_symlinked_destination(tmp_path, sync_script, target):
    (tmp_path / "site").mkdir()
    (tmp_path / "site/index.html").write_text("new docs")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "index.html").write_text("untouched")
    destination = tmp_path / "pages-repository" / target
    destination.parent.mkdir(parents=True)
    destination.symlink_to(outside, target_is_directory=True)

    result = sync_docs(tmp_path, sync_script)

    assert result.returncode != 0
    assert "Refusing to publish through a symlink" in result.stdout
    assert (outside / "index.html").read_text() == "untouched"

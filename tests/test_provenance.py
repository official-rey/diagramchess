"""Knowing which build is running, and replacing it.

Three reports in a row were an old install rather than a bug: the window
opened and showed last week's page with last week's script, and nothing on
screen or in the terminal said which build it was -- the version number does
not move between commits, so it could not have.  pip records the repository,
the branch and the resolved commit; these check that we read them back
correctly and can hand the right thing to pip again.
"""

import json

import pytest

from diagramchess.provenance import InstallSource, describe, install_source, version_line

BRANCH = "claude/chess-pdf-diagram-extractor-plh834"
FROM_GIT = {
    "url": "https://github.com/official-rey/diagramchess",
    "vcs_info": {"commit_id": "39c8833ad20d5693d388762d0936bd8619584d7a",
                 "requested_revision": BRANCH, "vcs": "git"},
}


def _installed_as(monkeypatch, direct_url):
    """Make importlib.metadata report this direct_url.json for the package."""
    from diagramchess import provenance

    class FakeDistribution:
        def read_text(self, name):
            return json.dumps(direct_url) if name == "direct_url.json" else None

    monkeypatch.setattr(provenance.metadata, "distribution",
                        lambda name: FakeDistribution())


def test_a_git_install_is_read_back_whole(monkeypatch):
    _installed_as(monkeypatch, FROM_GIT)
    source = install_source()
    assert source.vcs == "git"
    assert source.requested_revision == BRANCH
    assert source.short_commit == "39c8833"
    assert not source.editable


def test_the_requirement_asks_for_the_branch_not_the_commit():
    """Pinning the commit pip resolved would reinstall exactly what is already
    here, which is the one thing an update must not do."""
    requirement = InstallSource(**{
        "url": FROM_GIT["url"], "vcs": "git",
        "requested_revision": BRANCH,
        "commit_id": FROM_GIT["vcs_info"]["commit_id"],
    }).requirement()
    assert requirement == f"diagramchess @ git+{FROM_GIT['url']}@{BRANCH}"
    assert "39c8833" not in requirement


def test_a_commit_is_used_when_no_branch_was_named():
    source = InstallSource(url="https://example.invalid/x", vcs="git", commit_id="abc1234def")
    assert source.requirement().endswith("@abc1234def")


def test_the_version_line_names_the_build(monkeypatch):
    _installed_as(monkeypatch, FROM_GIT)
    assert version_line().endswith("(39c8833)")
    assert "39c8833" in describe() and BRANCH in describe()


def test_an_editable_checkout_says_so_rather_than_offering_an_update(monkeypatch):
    _installed_as(monkeypatch, {"url": "file:///home/someone/diagramchess",
                                "dir_info": {"editable": True}})
    assert install_source().editable
    assert version_line().endswith("(editable)")


@pytest.mark.parametrize("direct_url", [None, "", "not json at all", {"url": "x"}])
def test_a_package_with_no_usable_origin_does_not_raise(monkeypatch, direct_url):
    """A plain wheel records nothing, and this must degrade to the version."""
    from diagramchess import provenance

    class FakeDistribution:
        def read_text(self, name):
            if isinstance(direct_url, (dict, type(None))):
                return json.dumps(direct_url) if direct_url else None
            return direct_url

    monkeypatch.setattr(provenance.metadata, "distribution", lambda name: FakeDistribution())
    version_line()          # must not raise
    describe()


def test_update_refuses_to_pip_over_a_checkout(monkeypatch, capsys):
    """Reinstalling from git would silently replace the working tree someone
    is editing."""
    from diagramchess.cli import main

    _installed_as(monkeypatch, {"url": "file:///home/someone/diagramchess",
                                "dir_info": {"editable": True}})
    assert main(["update"]) == 1
    assert "git pull" in capsys.readouterr().err


def test_update_prints_the_pip_command_it_would_run(monkeypatch, capsys):
    from diagramchess.cli import main

    _installed_as(monkeypatch, FROM_GIT)
    assert main(["update", "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "-m pip install --upgrade --force-reinstall --no-deps" in printed
    # Without --no-deps this re-downloads most of a gigabyte of PyTorch that
    # has not changed; without --force-reinstall pip decides 0.1.0 == 0.1.0
    # and does nothing at all.
    assert f"diagramchess @ git+{FROM_GIT['url']}@{BRANCH}" in printed


def test_the_window_is_told_which_build_it_is(workspace):
    from fastapi.testclient import TestClient

    from diagramchess.review.app import create_app

    body = TestClient(create_app(workspace)).get("/api/stats").json()
    assert body["version"], "the header has nothing to show"

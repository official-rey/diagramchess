"""Where this copy came from, so that a stale one can say so.

Three separate reports have now been an old install rather than a bug: the
window opens, and what it shows is last week's page with last week's
JavaScript.  Nothing on screen said which version it was, and "0.1.0" would
not have helped if it had, because that number does not move between commits.

pip records rather more than the version.  A package installed from a git URL
carries a ``direct_url.json`` (PEP 610) naming the repository, the branch asked
for and the exact commit resolved -- enough to print something a person can
compare, and enough to install the same thing again without their having to
remember the URL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import metadata

PACKAGE = "diagramchess"


@dataclass(frozen=True)
class InstallSource:
    url: str
    vcs: str = ""
    requested_revision: str = ""
    commit_id: str = ""
    editable: bool = False

    @property
    def short_commit(self) -> str:
        return self.commit_id[:7]

    def requirement(self) -> str:
        """What to hand pip to install this same thing again."""
        if not self.vcs:
            return f"{PACKAGE} @ {self.url}"
        # The branch, not the commit: asking for a branch is what makes an
        # update an update rather than a reinstall of what is already here.
        revision = self.requested_revision or self.commit_id
        at = f"@{revision}" if revision else ""
        return f"{PACKAGE} @ {self.vcs}+{self.url}{at}"


def install_source() -> InstallSource | None:
    """None when pip recorded no origin -- a plain wheel, or a source tree."""
    try:
        raw = metadata.distribution(PACKAGE).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    vcs_info = data.get("vcs_info") or {}
    return InstallSource(
        url=data.get("url", ""),
        vcs=vcs_info.get("vcs", ""),
        requested_revision=vcs_info.get("requested_revision", ""),
        commit_id=vcs_info.get("commit_id", ""),
        editable=bool((data.get("dir_info") or {}).get("editable")),
    )


def version_line() -> str:
    """One short string naming this build, for a header or a --version."""
    from . import __version__

    source = install_source()
    if source is None:
        return __version__
    if source.editable:
        return f"{__version__} (editable)"
    if source.commit_id:
        return f"{__version__} ({source.short_commit})"
    return __version__


def describe() -> str:
    """The long form, for `dgc status` and for a bug report."""
    from . import __version__

    source = install_source()
    if source is None:
        return f"diagramchess {__version__} (installed from a package, no origin recorded)"
    if source.editable:
        return f"diagramchess {__version__} from a checkout at {source.url}"
    if source.vcs:
        return (f"diagramchess {__version__} from {source.url}\n"
                f"  branch {source.requested_revision or '(none named)'}"
                f" at commit {source.short_commit}")
    return f"diagramchess {__version__} from {source.url}"

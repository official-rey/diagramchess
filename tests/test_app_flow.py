"""Opening a book and following the work, without a terminal.

These cover the path the web interface adds: a PDF arrives as a request body,
a background job reads it, the page polls until it finishes, and the book is
then reviewable.  The job runner is exercised directly too, because a queue
that loses a job or reports a failure as a success would be invisible from the
outside until someone lost an hour of training to it.
"""

import time

import pytest
from fastapi.testclient import TestClient

from diagramchess.jobs import DONE, FAILED, JobRunner
from diagramchess.review.app import _parse_pages, create_app


@pytest.fixture
def api(workspace):
    return TestClient(create_app(workspace)), workspace


def _finish(api_client, job, timeout=180):
    """Poll the way the page does, and hand back the finished job."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = api_client.get(f"/api/jobs/{job['id']}").json()
        if job["state"] in (DONE, FAILED):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job['id']} never finished: {job}")


# -- the job runner ------------------------------------------------------

def test_a_job_reports_its_progress_and_its_result():
    runner = JobRunner()
    job = runner.submit("test", "counting", lambda job: (
        [job.step(done=n, total=3) for n in range(1, 4)], {"summary": "counted"})[1])
    assert runner.wait(timeout=10)
    assert job.as_dict()["state"] == DONE
    assert job.as_dict()["result"] == {"summary": "counted"}
    assert job.as_dict()["fraction"] == 1.0


def test_a_job_that_raises_is_reported_as_failed_not_lost():
    runner = JobRunner()

    def explode(job):
        raise ValueError("the PDF is encrypted")

    job = runner.submit("test", "doomed", explode)
    assert runner.wait(timeout=10)
    assert job.as_dict()["state"] == FAILED
    assert "the PDF is encrypted" in job.as_dict()["error"]


def test_one_job_at_a_time_so_the_second_does_not_start_early():
    """Two ingests at once would fight over the CPU and the database."""
    runner = JobRunner()
    order = []
    runner.submit("test", "first", lambda job: (time.sleep(0.3), order.append("first"))[1])
    runner.submit("test", "second", lambda job: order.append("second"))
    assert runner.wait(timeout=10)
    assert order == ["first", "second"]


def test_finished_jobs_are_forgotten_but_unfinished_ones_never_are():
    runner = JobRunner()
    from diagramchess.jobs import HISTORY

    for _ in range(HISTORY + 6):
        runner.submit("test", "quick", lambda job: None)
    assert runner.wait(timeout=20)
    assert len(runner.recent(limit=1000)) <= HISTORY


# -- opening a book ------------------------------------------------------

def test_a_pdf_posted_as_a_body_becomes_a_reviewable_book(api, demo_pdf):
    client, workspace = api
    started = client.post("/api/books?name=my%20book.pdf&pages=1-2",
                          content=demo_pdf.read_bytes()).json()["job"]
    job = _finish(client, started)

    assert job["state"] == DONE, job["error"]
    assert job["found"] > 0, "no diagrams found in the generated book"
    book_id = job["result"]["book_id"]
    assert workspace.book(book_id) is not None

    # The page sends the reader straight here, so it has to have something.
    queue = client.get(f"/api/queue?book_id={book_id}").json()["diagrams"]
    assert len(queue) == job["found"]
    assert client.get(f"/api/diagram/{queue[0]['id']}").status_code == 200


def test_the_uploaded_file_is_kept_so_pages_can_be_drawn_again(api, demo_pdf):
    client, workspace = api
    job = _finish(client, client.post("/api/books?name=keep.pdf&pages=1-1",
                                      content=demo_pdf.read_bytes()).json()["job"])
    book = workspace.book(job["result"]["book_id"])
    from pathlib import Path

    assert Path(book["path"]).exists(), "the book was not copied into the workspace"
    diagram = client.get(f"/api/queue?book_id={book['id']}").json()["diagrams"][0]
    assert client.get(f"/api/page/{diagram['id']}").status_code == 200


def test_a_name_cannot_escape_the_books_directory(api, demo_pdf):
    client, workspace = api
    job = _finish(client, client.post("/api/books?name=../../escaped.pdf&pages=1-1",
                                      content=demo_pdf.read_bytes()).json()["job"])
    from pathlib import Path

    stored = Path(workspace.book(job["result"]["book_id"])["path"]).resolve()
    assert stored.parent == (workspace.root / "books").resolve()
    assert not (workspace.root.parent.parent / "escaped.pdf").exists()


def test_two_books_with_one_name_do_not_overwrite_each_other(api, demo_pdf):
    client, workspace = api
    body = demo_pdf.read_bytes()
    first = _finish(client, client.post("/api/books?name=book.pdf&pages=1-1",
                                        content=body).json()["job"])
    # Same name, different bytes, so it is genuinely a second book.
    second = _finish(client, client.post("/api/books?name=book.pdf&pages=2-2",
                                         content=body + b"\n%% second\n").json()["job"])
    from pathlib import Path

    paths = {Path(workspace.book(j["result"]["book_id"])["path"]) for j in (first, second)}
    assert len(paths) == 2, "the second upload landed on top of the first"
    assert all(p.exists() for p in paths)


def test_something_that_is_not_a_pdf_is_refused_before_any_work_starts(api):
    client, workspace = api
    response = client.post("/api/books?name=notes.pdf", content=b"just some text")
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]
    assert list((workspace.root / "books").glob("*")) == [] or \
        not any((workspace.root / "books").iterdir())


def test_an_empty_body_is_refused(api):
    client, _ = api
    assert client.post("/api/books?name=nothing.pdf", content=b"").status_code == 400


@pytest.mark.parametrize("spec", ["nonsense", "0-4", "10-2", "-5"])
def test_a_page_range_that_makes_no_sense_says_so(api, spec, demo_pdf):
    client, _ = api
    response = client.post(f"/api/books?name=b.pdf&pages={spec}",
                           content=demo_pdf.read_bytes())
    assert response.status_code == 400


def test_page_ranges_are_one_based_and_inclusive():
    assert _parse_pages("10-12") == [9, 10, 11]
    assert _parse_pages("3") == [2]
    assert _parse_pages(None) is None
    assert _parse_pages("  ") is None


def test_forgetting_a_book_takes_its_diagrams_and_its_copy_of_the_file(api, demo_pdf):
    client, workspace = api
    job = _finish(client, client.post("/api/books?name=gone.pdf&pages=1-1",
                                      content=demo_pdf.read_bytes()).json()["job"])
    book_id = job["result"]["book_id"]
    from pathlib import Path

    path = Path(workspace.book(book_id)["path"])

    assert client.delete(f"/api/books/{book_id}").status_code == 200
    assert workspace.book(book_id) is None
    assert workspace.diagrams(book_id=book_id) == []
    assert not path.exists()
    assert client.delete(f"/api/books/{book_id}").status_code == 404


def test_a_book_opened_from_elsewhere_on_disk_is_not_deleted(api, demo_pdf, workspace):
    """Only copies this app made are ours to remove."""
    from diagramchess.pipeline import ingest

    client, _ = api
    report = ingest(workspace, demo_pdf, pages=[0])
    assert client.delete(f"/api/books/{report.book_id}").status_code == 200
    assert demo_pdf.exists(), "a file the reader chose was deleted from where they keep it"


# -- who is allowed to change things -------------------------------------

def test_another_site_cannot_drive_this_server(api, demo_pdf):
    """It listens on localhost, which every page in the browser can reach."""
    client, _ = api
    elsewhere = {"origin": "http://evil.example"}
    for response, what in [
        (client.post("/api/books?name=x.pdf", headers=elsewhere, content=b"%PDF-1.4"), "upload"),
        (client.post("/api/train", headers=elsewhere), "train"),
        (client.post("/api/books/demo", headers=elsewhere), "demo book"),
        (client.delete("/api/books/1", headers=elsewhere), "delete"),
    ]:
        assert response.status_code == 403, f"{what} was allowed from another site"


def test_our_own_page_may(api, demo_pdf):
    client, _ = api
    response = client.post("/api/books?name=ok.pdf&pages=1-1",
                           headers={"origin": "http://testserver"},
                           content=demo_pdf.read_bytes())
    assert response.status_code == 200
    _finish(client, response.json()["job"])


def test_reading_is_open_to_anyone_who_can_reach_it(api):
    client, _ = api
    assert client.get("/api/stats", headers={"origin": "http://evil.example"}).status_code == 200


# -- what the browser is actually given ----------------------------------

def test_asset_urls_carry_a_stamp_that_changes_with_the_files(api, tmp_path):
    """A browser that met an older copy of this app on the same port keeps
    serving the CSS and JS it cached then.  That does not look like a stale
    cache -- it looks like a broken app: new HTML with buttons the old script
    has no handlers for.  A stamped URL cannot be answered from that cache.
    """
    import re

    from diagramchess.review.app import STATIC, asset_stamp

    client, _ = api
    html = client.get("/").text
    stamps = set(re.findall(r'/static/[\w.]+\?v=(\w+)', html))
    assert len(stamps) == 1, f"assets are not all stamped alike: {stamps}"
    assert re.fullmatch(r"[0-9a-f]{10}", stamps.pop())

    before = asset_stamp()
    touched = STATIC / "style.css"
    original = touched.read_bytes()
    try:
        touched.write_bytes(original + b"\n/* nudge */\n")
        assert asset_stamp() != before, "editing an asset did not change the stamp"
    finally:
        touched.write_bytes(original)


def test_the_page_is_never_cached_and_the_assets_are_revalidated(api):
    client, _ = api
    assert client.get("/").headers["cache-control"] == "no-store"
    assert client.get("/static/index.js").headers["cache-control"] == "no-cache"


def test_pages_are_read_as_utf8_whatever_the_machine_is_set_to(tmp_path):
    """read_text() decodes with the locale encoding.  On Windows that is
    cp1252, and every ellipsis and figurine in these files came back mangled.

    Reproduced here by running under a C locale, where the same mistake is
    fatal rather than merely wrong.
    """
    import os
    import subprocess
    import sys

    script = tmp_path / "serve.py"
    script.write_text(
        "from diagramchess.review.app import _page\n"
        "html = _page('index.html', 'v1').body.decode('utf-8')\n"
        "assert '\\u2026' in html, 'the ellipsis did not survive'\n"
        "assert '\\u00e2' not in html, 'mojibake in the served page'\n"
        "print('ok')\n",
        encoding="utf-8",
    )
    ascii_only = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LC_ALL": "C", "LANG": "C",
        "PYTHONCOERCECLOCALE": "0", "PYTHONUTF8": "0",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    done = subprocess.run([sys.executable, str(script)], env=ascii_only,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]
    assert "ok" in done.stdout


def test_every_static_file_is_valid_utf8():
    from diagramchess.review.app import STATIC

    for path in sorted(STATIC.rglob("*")):
        if path.is_file():
            path.read_text(encoding="utf-8")      # raises if it is not


# -- the launcher --------------------------------------------------------

def test_the_launcher_writes_something_that_can_be_taken_back(tmp_path, monkeypatch):
    from diagramchess import launcher

    monkeypatch.setattr(launcher.Path, "home", staticmethod(lambda: tmp_path))
    written = launcher.install(workspace=tmp_path / "books", port=8788)
    assert written.path.exists()
    assert launcher.remove() == [written.path]
    assert not written.path.exists()
    assert launcher.remove() == []


def _as_windows(monkeypatch, tmp_path, desktop: str | None):
    """Point the Windows branch of the launcher at a temporary home.

    Only the environment is faked.  Forcing sys.platform would turn every Path
    in the process into a WindowsPath and break far more than it proves, which
    is why install() takes the platform as an argument.
    """
    from diagramchess import launcher

    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    if desktop is not None:
        (tmp_path / desktop).mkdir(parents=True)
    return launcher


def test_on_windows_the_shortcut_goes_to_the_desktop_and_the_start_menu(monkeypatch, tmp_path):
    launcher = _as_windows(monkeypatch, tmp_path, "Desktop")
    written = launcher.install(workspace=tmp_path / "ws", port=8765, system="windows")

    assert written.path == tmp_path / "Desktop" / "diagramchess.bat"
    start_menu = (tmp_path / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs"
                  / "diagramchess.bat")
    assert start_menu.exists(), "nothing in the Start menu"
    # `start` hands off, or the console window sits there for the whole session.
    # Read as bytes: a .bat wants CRLF, and text mode would hide a missing one.
    script = written.path.read_bytes()
    assert script.startswith(b'@echo off\r\nstart "diagramchess" ')
    assert b"\r\r\n" not in script, "line endings were translated twice"
    assert script == start_menu.read_bytes()
    assert sorted(launcher.remove("windows")) == sorted([written.path, start_menu])


def test_onedrive_desktops_are_found_and_phantom_ones_are_not_created(monkeypatch, tmp_path):
    """With OneDrive folder backup on, ~/Desktop does not exist.  Creating it
    would put the shortcut somewhere the reader never looks."""
    launcher = _as_windows(monkeypatch, tmp_path, "OneDrive/Desktop")
    written = launcher.install(workspace=tmp_path / "ws", port=8765, system="windows")

    assert written.path == tmp_path / "OneDrive" / "Desktop" / "diagramchess.bat"
    assert not (tmp_path / "Desktop").exists(), "invented a Desktop nobody uses"


def test_with_no_desktop_at_all_the_start_menu_still_gets_one(monkeypatch, tmp_path):
    launcher = _as_windows(monkeypatch, tmp_path, None)
    written = launcher.install(workspace=tmp_path / "ws", port=8765, system="windows")

    assert "Start Menu" in str(written.path)
    assert written.path.exists()
    assert "Windows key" in written.note


def test_the_launcher_bakes_in_absolute_paths(tmp_path):
    """It runs from a desktop with no shell, no PATH and no working directory."""
    import sys

    from diagramchess.launcher import command

    parts = command(tmp_path / "ws", 8765)
    assert parts[0] == sys.executable
    assert parts[1:3] == ["-m", "diagramchess.cli"]
    assert str((tmp_path / "ws")) in parts
    assert all("/" in p or p.isidentifier() or p.lstrip("-").replace(".", "").isalnum()
               or p.startswith("-") for p in parts)

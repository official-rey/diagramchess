"""Background work for the web interface.

Reading a book takes minutes and training takes the better part of an hour.
Neither can happen inside a request, and neither should require a terminal, so
they run here: one worker thread taking jobs off a queue, reporting progress
that a page can poll.

The queue is deliberately one job wide.  Ingesting and training are both
CPU-bound and both write to the same SQLite file; running two at once would
make each slower and the database busier, and the reader gains nothing from
starting a second book before the first is done.
"""

from __future__ import annotations

import queue
import threading
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"

# Enough history for the page to explain what happened this session without
# the list growing without bound behind a long-running server.
HISTORY = 24


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    """One unit of long work, and everything the page needs to draw it."""

    id: str
    kind: str
    label: str
    state: str = QUEUED
    total: int = 0
    done: int = 0
    found: int = 0
    note: str = ""
    result: dict = field(default_factory=dict)
    error: str = ""
    created_at: str = field(default_factory=_now)
    finished_at: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def step(self, *, done: int | None = None, total: int | None = None,
             found: int | None = None, note: str | None = None) -> None:
        """Called from the worker thread; every field is set under the lock so
        a poll arriving mid-update sees one consistent snapshot."""
        with self._lock:
            if done is not None:
                self.done = done
            if total is not None:
                self.total = total
            if found is not None:
                self.found = found
            if note is not None:
                self.note = note

    def as_dict(self) -> dict:
        with self._lock:
            fraction = self.done / self.total if self.total else 0.0
            return {
                "id": self.id, "kind": self.kind, "label": self.label,
                "state": self.state, "done": self.done, "total": self.total,
                "found": self.found, "note": self.note,
                "fraction": round(min(1.0, max(0.0, fraction)), 4),
                "result": dict(self.result), "error": self.error,
                "created_at": self.created_at, "finished_at": self.finished_at,
            }


class JobRunner:
    """A single worker thread and the jobs it has run."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _ensure_worker(self) -> None:
        # Started on first use rather than at import, so a process that only
        # reads diagrams never grows a thread it does not need.
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, name="dgc-jobs", daemon=True)
            self._thread.start()

    def submit(self, kind: str, label: str, work: Callable[[Job], dict | None],
               note: str = "") -> Job:
        # The note is set here rather than by the work itself because the first
        # thing a job does is import the machinery it needs, which can take
        # several seconds -- and a card that says nothing for several seconds
        # looks like a card that is not working.
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label, note=note)
        with self._lock:
            self._jobs[job.id] = job
            self._trim()
            self._ensure_worker()
        self._queue.put((job, work))
        return job

    def _trim(self) -> None:
        """Forget the oldest finished jobs, holding the lock.

        Jobs still queued or running are stepped over rather than stopping the
        trim: a burst of submissions is all unfinished at the moment it
        arrives, and a rule that gave up at the first of them would never
        forget anything at all.
        """
        while len(self._jobs) > HISTORY:
            stale = next((key for key, job in self._jobs.items()
                          if job.state in (DONE, FAILED)), None)
            if stale is None:
                return                 # nothing here is finished with yet
            self._jobs.pop(stale)

    def _work(self) -> None:
        while True:
            job, work = self._queue.get()
            job.state = RUNNING
            try:
                result = work(job)
                job.result = dict(result or {})
                job.state = DONE
            except Exception as exc:
                # The page is the only place this will ever be read, so the
                # message has to stand on its own; the traceback goes to the
                # console for whoever is in a position to care.
                job.error = f"{type(exc).__name__}: {exc}"
                job.state = FAILED
                traceback.print_exc()
            finally:
                job.finished_at = _now()
                with self._lock:
                    self._trim()
                self._queue.task_done()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 10) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())[-limit:][::-1]

    def active(self) -> Job | None:
        with self._lock:
            for job in reversed(self._jobs.values()):
                if job.state in (QUEUED, RUNNING):
                    return job
        return None

    def busy(self) -> bool:
        return self.active() is not None

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the queue drains; True if it did. For tests."""
        if timeout is None:
            self._queue.join()
            return True
        drained = threading.Event()
        threading.Thread(target=lambda: (self._queue.join(), drained.set()),
                         daemon=True).start()
        return drained.wait(timeout)

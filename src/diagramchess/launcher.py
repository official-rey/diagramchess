"""Desktop shortcuts, so the tool can be started without a terminal.

``dgc app`` already reduces the whole thing to one command, but one command is
still a terminal, a remembered incantation and a directory you have to be
standing in.  This writes a shortcut that carries all three: the absolute path
of the interpreter the tool is installed in, the workspace to open, and the
port.  After it, starting the tool is a double click.

Nothing here needs admin rights, and everything it writes is inside the user's
own home directory, so :func:`remove` can take it all back.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "diagramchess"


@dataclass
class Written:
    path: Path
    note: str


def command(workspace: Path, port: int) -> list[str]:
    """The command a shortcut runs.

    ``sys.executable -m diagramchess.cli`` rather than the ``dgc`` script, so
    the shortcut keeps working whether or not the virtual environment it was
    installed into is ever activated or on PATH.
    """
    return [sys.executable, "-m", "diagramchess.cli",
            "--workspace", str(workspace), "app", "--port", str(port)]


def this_system() -> str:
    if sys.platform == "darwin":
        return "macos"
    if os.name == "nt":
        return "windows"
    return "linux"


def install(workspace: Path, port: int = 8765, system: str | None = None) -> Written:
    """Write the shortcut for `system`, this machine's by default.

    The platform is an argument rather than something read from ``sys`` at the
    point of use, so the Windows path can be exercised from a test on any
    machine: faking it globally turns every ``Path`` in the process into a
    ``WindowsPath``, which breaks far more than it proves.
    """
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    writers = {"macos": _install_macos, "windows": _install_windows, "linux": _install_linux}
    return writers[system or this_system()](workspace, port)


def _windows_home() -> Path:
    return Path(os.environ.get("USERPROFILE") or Path.home())


def _windows_desktop() -> Path | None:
    """The Desktop, if it is where it can be found.

    With OneDrive's folder backup turned on -- the default on a good many
    machines -- the real Desktop is inside OneDrive and ``~/Desktop`` does not
    exist.  Creating it would put the shortcut in a folder nobody ever looks
    at, so an existing directory is the only thing accepted here.
    """
    home = _windows_home()
    for candidate in (home / "OneDrive" / "Desktop", home / "Desktop"):
        if candidate.is_dir():
            return candidate
    return None


def _windows_start_menu() -> Path:
    appdata = os.environ.get("APPDATA")
    root = Path(appdata) if appdata else _windows_home() / "AppData" / "Roaming"
    return root / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def paths(system: str | None = None) -> list[Path]:
    """Everywhere :func:`install` might have written, on this platform."""
    system = system or this_system()
    if system == "macos":
        return [Path.home() / "Applications" / f"{APP_NAME}.app"]
    if system == "windows":
        found = [_windows_start_menu() / f"{APP_NAME}.bat"]
        desktop = _windows_desktop()
        if desktop is not None:
            found.insert(0, desktop / f"{APP_NAME}.bat")
        return found
    return [Path.home() / ".local/share/applications" / f"{APP_NAME}.desktop"]


def remove(system: str | None = None) -> list[Path]:
    import shutil

    gone = []
    for path in paths(system):
        if path.is_dir():
            shutil.rmtree(path)
            gone.append(path)
        elif path.exists():
            path.unlink()
            gone.append(path)
    return gone


def _executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _install_linux(workspace: Path, port: int) -> Written:
    directory = Path.home() / ".local/share/applications"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{APP_NAME}.desktop"
    exec_line = " ".join(shlex.quote(part) for part in command(workspace, port))
    target.write_text(
        # The spec says UTF-8, and a home directory can have any name in it.
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        "Comment=Read chess diagrams out of PDF books\n"
        f"Exec={exec_line}\n"
        "Terminal=false\n"
        "Categories=Education;Game;\n",
        encoding="utf-8",
    )
    _executable(target)
    return Written(target, "It should appear in your applications menu; some desktops\n"
                           "  want a log out and back in first.")


def _install_macos(workspace: Path, port: int) -> Written:
    """A minimal .app bundle: a launch script and the plist that names it.

    A bare .command file would work too, but double-clicking one opens a
    Terminal window, which is the thing being got rid of.
    """
    bundle = Path.home() / "Applications" / f"{APP_NAME}.app"
    macos = bundle / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)

    script = macos / APP_NAME
    quoted = " ".join(shlex.quote(part) for part in command(workspace, port))
    script.write_text(f"#!/bin/sh\nexec {quoted}\n", encoding="utf-8")
    _executable(script)

    (bundle / "Contents" / "Info.plist").write_bytes(plistlib.dumps({
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": f"org.{APP_NAME}.app",
        "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleVersion": "1.0",
        # It is a server with a browser front end, so it has no windows of its
        # own and has no business taking over the Dock or the menu bar.
        "LSBackgroundOnly": True,
    }))
    return Written(bundle, "Open it from Applications, or drag it to the Dock.\n"
                           "  The first launch may need right-click → Open.")


def _install_windows(workspace: Path, port: int) -> Written:
    # pythonw.exe has no console; `start ""` hands the process off so the cmd
    # window the .bat runs in closes at once rather than sitting there for as
    # long as the tool is open.
    windowed = Path(sys.executable).with_name("pythonw.exe")
    runner = windowed if windowed.exists() else Path(sys.executable)
    parts = [str(runner), "-m", "diagramchess.cli",
             "--workspace", str(workspace), "app", "--port", str(port)]
    script = ("@echo off\r\n"
              'start "diagramchess" ' + " ".join(f'"{p}"' for p in parts) + "\r\n")

    # The Start menu is always there; the Desktop may not be.
    # newline="" so the CRLFs above are written exactly once: text mode would
    # otherwise translate the \n of each \r\n again and leave \r\r\n behind.
    start_menu = _windows_start_menu()
    start_menu.mkdir(parents=True, exist_ok=True)
    (start_menu / f"{APP_NAME}.bat").write_text(script, newline="")

    desktop = _windows_desktop()
    if desktop is None:
        return Written(start_menu / f"{APP_NAME}.bat",
                       "Press the Windows key and type diagramchess to find it.\n"
                       "  (No Desktop folder was found to put a copy on.)")
    target = desktop / f"{APP_NAME}.bat"
    target.write_text(script, newline="")
    return Written(target, "There is one in the Start menu too: press the Windows key\n"
                           "  and type diagramchess.")

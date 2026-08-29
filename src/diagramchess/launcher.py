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


def install(workspace: Path, port: int = 8765) -> Written:
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        return _install_macos(workspace, port)
    if os.name == "nt":
        return _install_windows(workspace, port)
    return _install_linux(workspace, port)


def paths() -> list[Path]:
    """Everywhere :func:`install` might have written, on this platform."""
    home = Path.home()
    if sys.platform == "darwin":
        return [home / "Applications" / f"{APP_NAME}.app"]
    if os.name == "nt":
        return [home / "Desktop" / f"{APP_NAME}.bat",
                home / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs" / f"{APP_NAME}.bat"]
    return [home / ".local/share/applications" / f"{APP_NAME}.desktop"]


def remove() -> list[Path]:
    import shutil

    gone = []
    for path in paths():
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
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        "Comment=Read chess diagrams out of PDF books\n"
        f"Exec={exec_line}\n"
        "Terminal=false\n"
        "Categories=Education;Game;\n"
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
    script.write_text(f"#!/bin/sh\nexec {quoted}\n")
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
    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    target = desktop / f"{APP_NAME}.bat"
    # pythonw.exe runs without a console window; fall back to python.exe if
    # this install has no windowed interpreter beside it.
    windowed = Path(sys.executable).with_name("pythonw.exe")
    runner = windowed if windowed.exists() else Path(sys.executable)
    parts = [str(runner), "-m", "diagramchess.cli",
             "--workspace", str(workspace), "app", "--port", str(port)]
    target.write_text("@echo off\r\n" + " ".join(f'"{p}"' for p in parts) + "\r\n")
    return Written(target, "It is on your Desktop; right-click → Pin to Start if you\n"
                           "  would like it in the Start menu too.")

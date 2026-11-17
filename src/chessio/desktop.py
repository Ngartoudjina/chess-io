"""Intégration au bureau Ubuntu : lanceur dans le menu des applications et icône.

``chessio installer`` écrit, pour l'utilisateur courant (sans droits
administrateur), le lanceur ``~/.local/share/applications/chessio.desktop`` et
l'icône dans le thème d'icônes ; ``chessio desinstaller`` les retire.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

LOGO = Path(__file__).with_name("ui") / "assets" / "chessio.svg"


def data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def desktop_file() -> Path:
    return data_home() / "applications" / "chessio.desktop"


def icon_file() -> Path:
    return data_home() / "icons" / "hicolor" / "scalable" / "apps" / "chessio.svg"


def launcher_command() -> str:
    """Commande lançant l'interface avec l'environnement Python courant."""
    gui = Path(sys.executable).with_name("chessio-gui")
    if gui.exists():
        return str(gui)
    return f"{sys.executable} -m chessio gui"


def desktop_entry(command: str) -> str:
    return "\n".join(
        [
            "[Desktop Entry]",
            "Type=Application",
            "Version=1.5",
            "Name=Chessio",
            "GenericName=Assistant d'échecs",
            "Comment=Le meilleur coup à partir d'une capture d'échiquier ; jouer contre l'IA",
            f"Exec={command}",
            "Icon=chessio",
            "Terminal=false",
            "Categories=Game;BoardGame;",
            "Keywords=échecs;chess;stockfish;analyse;capture;",
            "StartupNotify=true",
            "StartupWMClass=chessio",
            "Actions=jouer;",
            "",
            "[Desktop Action jouer]",
            "Name=Jouer contre l'IA",
            f"Exec={command.replace('chessio-gui', 'chessio')} jouer"
            if command.endswith("chessio-gui")
            else f"Exec={command.removesuffix(' gui')} jouer",
            "",
        ]
    )


def _refresh() -> None:
    for tool, arg in (
        ("update-desktop-database", desktop_file().parent),
        ("gtk-update-icon-cache", data_home() / "icons" / "hicolor"),
    ):
        if shutil.which(tool):
            subprocess.run([tool, "-q", str(arg)], check=False, capture_output=True)


def install() -> list[Path]:
    desktop_file().parent.mkdir(parents=True, exist_ok=True)
    icon_file().parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(LOGO, icon_file())
    desktop_file().write_text(desktop_entry(launcher_command()), encoding="utf-8")
    desktop_file().chmod(0o755)
    _refresh()
    return [desktop_file(), icon_file()]


def uninstall() -> list[Path]:
    removed = []
    for path in (desktop_file(), icon_file()):
        if path.exists():
            path.unlink()
            removed.append(path)
    _refresh()
    return removed

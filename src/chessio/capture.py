"""Captures à la demande : téléphone Android branché en USB, ou écran de l'ordinateur.

Chaque capture est déclenchée par l'utilisateur (un clic, une capture) : rien ne
tourne en arrière-plan. Usage prévu : parties contre les bots, problèmes, parties
terminées (cahier des charges, section 2).

Téléphone : ``adb exec-out screencap -p`` (débogage USB activé sur le téléphone).
Écran : voir ``chessio.ui.screen`` (portail XDG, seul accès autorisé sous Wayland).
"""

from __future__ import annotations

import io
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
from PIL import Image


class CaptureError(RuntimeError):
    """Capture impossible (téléphone débranché, débogage USB refusé…)."""


@dataclass(frozen=True)
class Phone:
    serial: str
    state: str  # « device » : prêt ; « unauthorized » : autorisation à donner sur le téléphone
    model: str

    @property
    def ready(self) -> bool:
        return self.state == "device"


def adb_path() -> str | None:
    return shutil.which("adb")


def phones() -> list[Phone]:
    """Téléphones Android vus par adb (liste vide si adb est absent)."""
    adb = adb_path()
    if adb is None:
        return []
    try:
        out = subprocess.run(
            [adb, "devices", "-l"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        model = next((p.split(":", 1)[1] for p in parts[2:] if p.startswith("model:")), "")
        found.append(Phone(parts[0], parts[1], model.replace("_", " ")))
    return found


def capture_phone(serial: str) -> np.ndarray:
    """Capture d'écran du téléphone, en RGB."""
    adb = adb_path()
    if adb is None:
        raise CaptureError("adb est introuvable (paquet android-tools-adb).")
    try:
        result = subprocess.run(
            [adb, "-s", serial, "exec-out", "screencap", "-p"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptureError(f"Le téléphone ne répond pas : {exc}") from exc
    if result.returncode != 0 or not result.stdout.startswith(b"\x89PNG"):
        message = result.stderr.decode(errors="replace").strip() or "réponse inattendue"
        raise CaptureError(f"Capture du téléphone impossible : {message}")
    with Image.open(io.BytesIO(result.stdout)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

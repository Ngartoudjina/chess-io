"""Emplacements des données locales (norme XDG)."""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "chessio"


def templates_path() -> Path:
    """Gabarits issus de la calibration (lot L1)."""
    return data_dir() / "gabarits.npz"

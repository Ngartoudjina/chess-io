"""Mises à jour : dernière version publiée sur GitHub (Releases), et remplacement de l'AppImage.

Aucun accès réseau sans demande (ENF-08) : la vérification se fait depuis le menu,
ou au démarrage si l'utilisateur l'a activée. La requête ne transmet rien de
personnel (agent générique, aucune donnée de l'utilisateur).

Le remplacement est atomique : la nouvelle version est téléchargée à côté de
l'AppImage, rendue exécutable, puis substituée d'un seul coup (``os.replace``).
En cas d'échec, la version en place reste intacte.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from chessio import __version__

REPOSITORY = "Ngartoudjina/chess-io"
LATEST = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPOSITORY}/releases/latest"
USER_AGENT = "Chessio-mise-a-jour"


class UpdateError(RuntimeError):
    """Vérification ou installation de la mise à jour impossible."""


@dataclass(frozen=True)
class Release:
    version: str
    notes: str
    page: str
    asset_url: str | None  # AppImage de cette version, si publiée
    asset_size: int


def parse_version(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.strip().lstrip("vV").split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def is_newer(version: str, current: str = __version__) -> bool:
    return parse_version(version) > parse_version(current)


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def latest_release(timeout: float = 15) -> Release:
    try:
        with urllib.request.urlopen(_request(LATEST), timeout=timeout) as response:
            data = json.load(response)
    except OSError as exc:
        raise UpdateError(f"GitHub injoignable : {exc}") from exc
    asset = next(
        (a for a in data.get("assets", []) if a.get("name", "").endswith(".AppImage")), None
    )
    return Release(
        version=str(data.get("tag_name", "0")).lstrip("vV"),
        notes=str(data.get("body") or ""),
        page=str(data.get("html_url") or RELEASES_PAGE),
        asset_url=asset.get("browser_download_url") if asset else None,
        asset_size=int(asset.get("size", 0)) if asset else 0,
    )


def appimage_path() -> Path | None:
    """Fichier AppImage en cours d'exécution (``None`` : version de développement)."""
    path = os.environ.get("APPIMAGE")
    return Path(path) if path else None


def install_update(
    release: Release,
    target: Path,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    timeout: float = 60,
) -> Path:
    """Télécharge l'AppImage de ``release`` et remplace ``target`` ; renvoie ``target``."""
    if release.asset_url is None:
        raise UpdateError("Cette version ne propose pas encore d'AppImage à télécharger.")
    partial = target.with_name(target.name + ".telechargement")
    try:
        with urllib.request.urlopen(_request(release.asset_url), timeout=timeout) as response:
            total = int(response.headers.get("Content-Length") or release.asset_size or 0)
            done = 0
            with partial.open("wb") as out:
                while chunk := response.read(256 * 1024):
                    if should_stop is not None and should_stop():
                        raise UpdateError("Mise à jour annulée.")
                    out.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
        if total and partial.stat().st_size != total:
            raise UpdateError("Téléchargement incomplet : réessayez.")
        with partial.open("rb") as check:
            if check.read(4) != b"\x7fELF":
                raise UpdateError("Le fichier téléchargé n'est pas une AppImage valide.")
        partial.chmod(0o755)
        os.replace(partial, target)
    except OSError as exc:
        raise UpdateError(f"Téléchargement impossible : {exc}") from exc
    finally:
        if partial.exists():
            partial.unlink()
    return target

"""Mises à jour : comparaison des versions et remplacement atomique de l'AppImage."""

import pytest

from chessio.update import Release, UpdateError, install_update, is_newer, parse_version


def test_versions():
    assert parse_version("v0.10.2") == (0, 10, 2)
    assert is_newer("0.2.0", "0.1.0")
    assert is_newer("v1.0", "0.9.9")
    assert not is_newer("0.1.0", "0.1.0")
    assert not is_newer("0.0.9", "0.1.0")


def release_for(path) -> Release:
    return Release("9.9.9", "notes", "page", path.as_uri(), path.stat().st_size)


def test_replaces_appimage_atomically(tmp_path):
    new = tmp_path / "nouvelle.AppImage"
    new.write_bytes(b"\x7fELF nouvelle version")
    target = tmp_path / "Chessio.AppImage"
    target.write_bytes(b"\x7fELF ancienne version")
    seen = []
    install_update(release_for(new), target, progress=lambda d, t: seen.append((d, t)))
    assert target.read_bytes() == b"\x7fELF nouvelle version"
    assert target.stat().st_mode & 0o111  # exécutable
    assert seen[-1][0] == seen[-1][1]
    assert not list(tmp_path.glob("*.telechargement"))


def test_corrupt_download_keeps_current_version(tmp_path):
    bad = tmp_path / "page_html.AppImage"
    bad.write_bytes(b"<html>erreur</html>")
    target = tmp_path / "Chessio.AppImage"
    target.write_bytes(b"\x7fELF version en place")
    with pytest.raises(UpdateError):
        install_update(release_for(bad), target)
    assert target.read_bytes() == b"\x7fELF version en place"


def test_release_without_appimage():
    with pytest.raises(UpdateError):
        install_update(Release("1.0", "", "", None, 0), __import__("pathlib").Path("x"))

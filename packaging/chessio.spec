# Recette PyInstaller : Python, l'interface, les réseaux, le logo et les deux moteurs.
# Utilisée par packaging/construire_appimage.sh (lot L6).
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).parent
engines = [
    (str(ROOT / "engines" / "stockfish"), "engines"),
    (str(ROOT / "moteur-rs" / "target" / "release" / "chessio-moteur"), "engines"),
]

a = Analysis(
    [str(ROOT / "packaging" / "lanceur.py")],
    pathex=[str(ROOT / "src")],
    binaries=[(src, dest) for src, dest in engines if Path(src).exists()],
    datas=collect_data_files("chessio") + collect_data_files("chess"),
    hiddenimports=["chessio.moteur.uci", "PySide6.QtSvg", "PySide6.QtDBus"],
    excludes=["tkinter", "pytest", "ruff"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="chessio", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="chessio")

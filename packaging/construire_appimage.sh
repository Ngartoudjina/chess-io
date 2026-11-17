#!/usr/bin/env bash
# Construit dist/Chessio-<version>-x86_64.AppImage : un seul fichier, autonome.
# Prérequis : Stockfish (scripts/fetch_stockfish.sh), moteur Rust compilé
# (cargo build --release --manifest-path moteur-rs/Cargo.toml), PyInstaller.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
VERSION="$("$PYTHON" -c 'import chessio; print(chessio.__version__)')"
APPDIR="build/Chessio.AppDir"
TOOL="build/appimagetool"

"$PYTHON" -m PyInstaller --noconfirm --clean --distpath build/pyinstaller --workpath build/travail packaging/chessio.spec

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/lib"
cp -a build/pyinstaller/chessio "$APPDIR/usr/lib/chessio"
cp src/chessio/ui/assets/chessio.svg "$APPDIR/chessio.svg"
cat > "$APPDIR/chessio.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Chessio
GenericName=Assistant d'échecs
Comment=Le meilleur coup à partir d'une capture d'échiquier ; jouer contre l'IA
Exec=chessio
Icon=chessio
Terminal=false
Categories=Game;BoardGame;
StartupWMClass=chessio
X-AppImage-Version=$VERSION
DESKTOP
cat > "$APPDIR/AppRun" <<'APPRUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/lib/chessio/chessio" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

if [ ! -x "$TOOL" ]; then
    curl --fail --location --retry 5 -o "$TOOL" \
        https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$TOOL"
fi
mkdir -p dist
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$TOOL" --no-appstream "$APPDIR" "dist/Chessio-$VERSION-x86_64.AppImage"
echo "Paquet : dist/Chessio-$VERSION-x86_64.AppImage"

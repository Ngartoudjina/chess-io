#!/usr/bin/env bash
# Télécharge Stockfish (binaire Linux x86-64 « universal », qui choisit seul
# le jeu d'instructions du processeur) dans engines/stockfish, après
# vérification de l'empreinte SHA-256 publiée sur GitHub.
set -euo pipefail

VERSION="sf_19"
ASSET="stockfish-linux-x86-64-universal.tar.gz"
SHA256="9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f"
URL="https://github.com/official-stockfish/Stockfish/releases/download/${VERSION}/${ASSET}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/engines"
TARGET="${DEST}/stockfish"

if [[ -x "${TARGET}" && "${1:-}" != "--force" ]]; then
    echo "Stockfish déjà présent : ${TARGET} (--force pour le retélécharger)"
    exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# STOCKFISH_ARCHIVE=/chemin/archive.tar.gz évite de retélécharger (connexion lente).
if [[ -n "${STOCKFISH_ARCHIVE:-}" ]]; then
    cp "${STOCKFISH_ARCHIVE}" "${WORK}/${ASSET}"
else
    echo "Téléchargement de ${ASSET} (${VERSION})…"
    curl --fail --location --retry 5 --retry-all-errors --output "${WORK}/${ASSET}" "${URL}"
fi
echo "${SHA256}  ${WORK}/${ASSET}" | sha256sum --check --quiet

tar -xzf "${WORK}/${ASSET}" -C "${WORK}"
BINARY="$(find "${WORK}" -type f -name 'stockfish*' ! -name '*.tar.gz' -perm -u+x | head -n 1)"
if [[ -z "${BINARY}" ]]; then
    echo "Exécutable introuvable dans l'archive" >&2
    exit 1
fi

mkdir -p "${DEST}"
install -m 755 "${BINARY}" "${TARGET}"
echo "Stockfish installé : ${TARGET}"
"${TARGET}" <<< "quit" | head -n 1

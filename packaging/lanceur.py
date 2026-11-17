"""Point d'entrée du paquet autonome (PyInstaller) : la commande « chessio »."""

import sys

from chessio.__main__ import main

if __name__ == "__main__":
    sys.exit(main())

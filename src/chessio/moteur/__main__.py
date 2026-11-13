"""Point d'entrée UCI : ``python -m chessio.moteur``."""

import sys

from chessio.moteur.uci import main

if __name__ == "__main__":
    sys.exit(main())

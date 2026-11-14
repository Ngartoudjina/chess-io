"""Classification des 64 cases : réseau convolutif (lot L2), à défaut gabarits (lot L1).

Le réseau reconnaît les pièces de n'importe quel thème sans calibration. Les
gabarits, appris sur une capture de la position initiale, ne servent plus que
si le modèle est absent.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from chessio.vision.cnn import LABELS, MODEL_PATH, UNKNOWN, TileNet, features
from chessio.vision.templates import CellPrediction, TemplateSet

# Probabilité en deçà de laquelle une case est signalée comme douteuse (EF-07).
# Réglée sur les captures réelles : voir scripts/entrainer_cases.py.
CNN_MIN_CONFIDENCE = 0.80


class NoClassifierError(RuntimeError):
    """Ni modèle entraîné, ni calibration : impossible de lire les cases."""


@lru_cache(maxsize=1)
def load_model() -> TileNet | None:
    return TileNet.load(MODEL_PATH) if MODEL_PATH.exists() else None


def classify_cells(
    cells: np.ndarray, templates: TemplateSet | None = None, net: TileNet | None = None
) -> list[list[CellPrediction]]:
    """Cases ``(8, 8, 64, 64, 3)`` en ordre de l'écran → prédictions ligne par ligne."""
    net = net if net is not None else load_model()
    if net is not None:
        probs = net.predict(features(cells.reshape(-1, 64, 64, 3)))
        predictions = []
        for p in probs:
            label = LABELS[int(p.argmax())]
            if label == UNKNOWN:  # objet étranger : case laissée vide, à vérifier
                predictions.append(CellPrediction(".", 0.0, threshold=CNN_MIN_CONFIDENCE))
            else:
                predictions.append(
                    CellPrediction(label, float(p.max()), threshold=CNN_MIN_CONFIDENCE)
                )
        return [predictions[row * 8 : (row + 1) * 8] for row in range(8)]
    if templates is not None:
        return templates.classify(cells)
    raise NoClassifierError(
        "Aucun modèle de reconnaissance (chessio/vision/modeles/cases.npz) ni calibration."
    )

"""Réseau des cases (NumPy) : la rétropropagation doit égaler les dérivées numériques."""

import numpy as np
import pytest

from chessio.vision.cnn import CHANNELS, LABELS, MODEL_PATH, TileNet, features


def as_float64(net: TileNet) -> TileNet:
    return TileNet({k: v.astype(np.float64) for k, v in net.params.items()})


def test_gradients_match_finite_differences():
    net = as_float64(TileNet.initialize(seed=1))
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 32, 32, CHANNELS))
    y = np.array([0, 5, 12])
    _, grads, _ = net.loss_and_gradients(x, y)
    eps = 1e-6
    for name, value in net.params.items():
        for _ in range(4):
            index = tuple(int(rng.integers(s)) for s in value.shape)
            saved = value[index]
            value[index] = saved + eps
            plus, _, _ = net.loss_and_gradients(x, y)
            value[index] = saved - eps
            minus, _, _ = net.loss_and_gradients(x, y)
            value[index] = saved
            numeric = (plus - minus) / (2 * eps)
            analytic = grads[name][index]
            assert abs(numeric - analytic) <= 1e-5 + 1e-3 * abs(numeric), (name, index)


def test_features_shape_and_background_invariance():
    cells = np.zeros((2, 64, 64, 3), np.uint8)
    cells[0] = (235, 236, 208)
    cells[1] = (115, 149, 82)
    feats = features(cells)
    assert feats.shape == (2, 32, 32, CHANNELS)
    # Case vide : l'écart au fond est nul, quelle que soit sa couleur.
    assert np.allclose(feats[..., 3:], 0)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="modèle non entraîné")
def test_saved_model_loads_and_predicts():
    net = TileNet.load()
    probs = net.predict(features(np.full((4, 64, 64, 3), 200, np.uint8)))
    assert probs.shape == (4, len(LABELS))
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)

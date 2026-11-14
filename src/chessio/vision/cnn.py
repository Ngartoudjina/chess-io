"""Réseau convolutif de classification des cases (lot L2, EF-05), en NumPy.

Treize classes : case vide, et les six pièces de chaque couleur. Le réseau est
petit (≈ 95 000 poids) : une couche d'entrée qui projette chaque bloc 2 × 2,
deux convolutions 3 × 3 suivies chacune d'un max-pooling, puis deux couches
denses. Il s'entraîne en quelques minutes
sur le processeur et classe les 64 cases d'un plateau en ~10 ms.

Entrée : la case normalisée (64 × 64) réduite en 32 × 32, sur 6 canaux — les
couleurs, et leur écart à la couleur de fond de la case. Ce second groupe rend
la lecture indépendante des couleurs du plateau.

Écrit en NumPy plutôt qu'avec PyTorch (cahier des charges, section 6) : la
connexion du poste rend un téléchargement de 200 Mo impraticable, et l'inférence
n'ajoute ainsi aucune dépendance à l'application.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chessio.vision.ops import background

LABELS = (".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k", "?")
UNKNOWN = "?"  # objet étranger (curseur, fenêtre, emoji…) : case à vérifier
INPUT_SIDE = 32
CHANNELS = 6
MODEL_PATH = Path(__file__).with_name("modeles") / "cases.npz"


def features(cells: np.ndarray) -> np.ndarray:
    """Cases ``(k, 64, 64, 3)`` (uint8) → entrées du réseau ``(k, 32, 32, 6)`` (float32)."""
    cells = cells.reshape(-1, 64, 64, 3)
    reduced = cells.reshape(-1, 32, 2, 32, 2, 3).mean(axis=(2, 4), dtype=np.float32) / 255
    return features_from_reduced(reduced, background(cells) / 255)


def features_from_reduced(reduced: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Entrées à partir des cases déjà réduites (``(k, 32, 32, 3)`` dans [0, 1]) et du fond."""
    reduced = reduced.astype(np.float32)
    relative = reduced - bg.astype(np.float32)[:, None, None, :]
    return np.concatenate([reduced - 0.5, relative], axis=-1)


# --- couches -----------------------------------------------------------------


def conv_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convolution 3 × 3 de pas 1, bords complétés par des zéros. ``x`` : (N, H, W, C)."""
    n, h, wd, c = x.shape
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    # Colonnes « im2col » : les 9 voisinages décalés, côte à côte (ordre i, j, canal).
    cols = np.concatenate(
        [padded[:, i : i + h, j : j + wd, :] for i in range(3) for j in range(3)], axis=-1
    ).reshape(n * h * wd, 9 * c)
    out = cols @ w.reshape(9 * c, -1) + b
    return out.reshape(n, h, wd, -1), cols


def conv_backward(
    dout: np.ndarray, cols: np.ndarray, x_shape: tuple, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, h, wd, c = x_shape
    d = dout.reshape(-1, dout.shape[-1])
    dw = (cols.T @ d).reshape(w.shape)
    db = d.sum(axis=0)
    # Gradient de l'entrée : un produit matriciel par position du noyau, ajouté à
    # la fenêtre correspondante (plus rapide que de redistribuer les colonnes).
    dx = np.zeros((n, h + 2, wd + 2, c), dtype=dout.dtype)
    for i in range(3):
        for j in range(3):
            dx[:, i : i + h, j : j + wd, :] += (d @ w[i, j].T).reshape(n, h, wd, c)
    return dx[:, 1:-1, 1:-1, :], dw, db


def patch_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Couche d'entrée « patchify » : chaque bloc 2 × 2 (sans recouvrement) devient un
    vecteur, projeté linéairement. Simple réarrangement : aucune copie de fenêtres."""
    n, h, wd, c = x.shape
    cols = (
        x.reshape(n, h // 2, 2, wd // 2, 2, c)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(n * (h // 2) * (wd // 2), 4 * c)
    )
    out = cols @ w.reshape(4 * c, -1) + b
    return out.reshape(n, h // 2, wd // 2, -1), cols


def patch_backward(
    dout: np.ndarray, cols: np.ndarray, x_shape: tuple, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n, h, wd, c = x_shape
    d = dout.reshape(-1, dout.shape[-1])
    return (cols.T @ d).reshape(w.shape), d.sum(axis=0)


def pool_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Max-pooling 2 × 2."""
    n, h, w, c = x.shape
    blocks = x.reshape(n, h // 2, 2, w // 2, 2, c)
    out = blocks.max(axis=(2, 4))
    mask = blocks == out[:, :, None, :, None, :]
    return out, mask


def pool_backward(dout: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n, h, _, w, _, c = mask.shape
    return (mask * dout[:, :, None, :, None, :]).reshape(n, h * 2, w * 2, c)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# --- modèle --------------------------------------------------------------------


@dataclass
class TileNet:
    params: dict[str, np.ndarray]

    @staticmethod
    def shapes(classes: int) -> dict[str, tuple[int, ...]]:
        return {
            "s1": (2, 2, CHANNELS, 24),  # entrée 32 × 32 → 16 × 16 × 24
            "c2": (3, 3, 24, 32),  # 16 × 16 → (max-pooling) 8 × 8 × 32
            "c3": (3, 3, 32, 64),  # 8 × 8 → (max-pooling) 4 × 4 × 64
            "d1": (4 * 4 * 64, 64),
            "d2": (64, classes),
        }

    @classmethod
    def initialize(cls, seed: int = 0, classes: int = len(LABELS)) -> TileNet:
        """Poids aléatoires (initialisation de He), biais nuls."""
        rng = np.random.default_rng(seed)
        params = {}
        for name, shape in cls.shapes(classes).items():
            fan_in = int(np.prod(shape[:-1]))
            params[name] = (rng.standard_normal(shape) * np.sqrt(2 / fan_in)).astype(np.float32)
            params[name + "b"] = np.zeros(shape[-1], dtype=np.float32)
        return cls(params)

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> TileNet:
        with np.load(Path(path)) as data:
            return cls({name: data[name].astype(np.float32) for name in data.files})

    def save(self, path: str | Path = MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.params)

    def forward(self, x: np.ndarray, keep: bool = False):
        """Logits ``(N, 13)`` ; avec ``keep``, aussi les valeurs utiles à la rétropropagation."""
        p = self.params
        cache = {}
        z0, cols0 = patch_forward(x, p["s1"], p["s1b"])
        h = np.maximum(z0, 0)
        if keep:
            cache["s1"] = (x.shape, cols0, z0)
        for name in ("c2", "c3"):
            z, cols = conv_forward(h, p[name], p[name + "b"])
            a = np.maximum(z, 0)
            pooled, mask = pool_forward(a)
            if keep:
                cache[name] = (h.shape, cols, z, mask)
            h = pooled
        flat = h.reshape(h.shape[0], -1)
        z1 = flat @ p["d1"] + p["d1b"]
        a1 = np.maximum(z1, 0)
        logits = a1 @ p["d2"] + p["d2b"]
        if keep:
            cache["flat"] = (h.shape, flat)
            cache["d1"] = (z1, a1)
        return (logits, cache) if keep else logits

    def predict(self, x: np.ndarray, batch: int = 256) -> np.ndarray:
        """Probabilités ``(N, 13)``."""
        return np.concatenate(
            [softmax(self.forward(x[i : i + batch])) for i in range(0, len(x), batch)]
        )

    def loss_and_gradients(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray], np.ndarray]:
        """Entropie croisée moyenne, gradients des poids, probabilités."""
        p = self.params
        logits, cache = self.forward(x, keep=True)
        probs = softmax(logits)
        n = len(y)
        loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())
        grads: dict[str, np.ndarray] = {}
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1
        dlogits /= n
        z1, a1 = cache["d1"]
        pooled_shape, flat = cache["flat"]
        grads["d2"] = a1.T @ dlogits
        grads["d2b"] = dlogits.sum(axis=0)
        da1 = dlogits @ p["d2"].T
        dz1 = da1 * (z1 > 0)
        grads["d1"] = flat.T @ dz1
        grads["d1b"] = dz1.sum(axis=0)
        dh = (dz1 @ p["d1"].T).reshape(pooled_shape)
        for name in ("c3", "c2"):
            x_shape, cols, z, mask = cache[name]
            da = pool_backward(dh, mask)
            dz = da * (z > 0)
            dh, grads[name], grads[name + "b"] = conv_backward(dz, cols, x_shape, p[name])
        x_shape, cols0, z0 = cache["s1"]
        grads["s1"], grads["s1b"] = patch_backward(dh * (z0 > 0), cols0, x_shape, p["s1"])
        return loss, grads, probs


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 1e-3) -> None:
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            m_hat = self.m[k] / (1 - b1**self.t)
            v_hat = self.v[k] / (1 - b2**self.t)
            params[k] -= (self.lr * m_hat / (np.sqrt(v_hat) + eps)).astype(params[k].dtype)

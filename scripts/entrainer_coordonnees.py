"""Entraînement du réseau qui lit les coordonnées des cases (lot L2, orientation).

Les chiffres et lettres sont dessinés dans toutes les polices latines installées
(couleurs, contrastes, tailles et positions variés), dans le coin d'une case
synthétique. Validation sur les coordonnées des captures réelles, dont les
polices ne servent jamais à l'entraînement.

    python scripts/entrainer_coordonnees.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import chess  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrainer_cases import PHONE, START, chesscom_captures  # noqa: E402

from chessio.vision.board import CELL, screen_to_square  # noqa: E402
from chessio.vision.cnn import Adam, TileNet  # noqa: E402
from chessio.vision.coords import (  # noqa: E402
    DIGIT_ZONE,
    GLYPHS,
    LETTER_ZONE,
    MODEL_PATH,
    corner_features,
)
from chessio.vision.dataset import capture_cells, default_sets, degrade, random_theme  # noqa: E402

EXCLUDED = ("symbol", "math", "emoji", "cjk", "mono", "devanagari", "arabic", "hebrew", "thai")


def latin_fonts() -> list[str]:
    """Polices installées couvrant le français (hors symboles, idéogrammes…)."""
    out = subprocess.run(
        ["fc-list", ":lang=fr", "file"], capture_output=True, text=True, check=False
    ).stdout
    files = sorted({line.split(":")[0].strip() for line in out.splitlines() if line.strip()})
    return [f for f in files if not any(word in f.lower() for word in EXCLUDED)]


def synth(rng: np.random.Generator, fonts: list[str], sets: list) -> tuple[np.ndarray, int, int]:
    """Une case synthétique, la zone examinée (0 : chiffre, 1 : lettre) et le caractère."""
    light, dark = random_theme(rng)
    own, other = (light, dark) if rng.random() < 0.5 else (dark, light)
    image = Image.new("RGB", (CELL, CELL), own)
    if rng.random() < 0.4:  # une pièce au centre, qui déborde parfois vers le coin
        pieces = sets[int(rng.integers(len(sets)))]
        symbol = list(pieces.sprites)[int(rng.integers(len(pieces.sprites)))]
        sprite = Image.fromarray(pieces.sprites[symbol][0])
        size = int(CELL * rng.uniform(0.8, 1.05))
        sprite = sprite.resize((size, size), Image.Resampling.LANCZOS)
        offset = (CELL - size) // 2
        image.paste(sprite, (offset, offset), sprite)
    zone = int(rng.integers(2))
    label = 0
    if rng.random() < 0.8:
        label = int(rng.integers(1, 9)) if zone == 0 else int(rng.integers(9, 17))
        draw = ImageDraw.Draw(image, "RGBA")
        font = ImageFont.truetype(fonts[int(rng.integers(len(fonts)))], int(rng.integers(8, 19)))
        kind = rng.random()
        if kind < 0.45:
            ink = other
        elif kind < 0.75:
            ink = tuple(int(v) for v in rng.integers(90, 190, 1).repeat(3))  # gris
        else:
            ink = tuple(int(v) for v in rng.integers(0, 256, 3))
        alpha = int(rng.uniform(0.45, 1.0) * 255)
        if zone == 0:
            x, y = int(rng.integers(1, 8)), int(rng.integers(0, 7))
            draw.text((x, y), GLYPHS[label], font=font, fill=ink + (alpha,), anchor="lt")
        else:
            x, y = CELL - int(rng.integers(1, 8)), CELL - int(rng.integers(0, 6))
            draw.text((x, y), GLYPHS[label], font=font, fill=ink + (alpha,), anchor="rb")
    return degrade(rng, np.asarray(image, dtype=np.uint8)), zone, label


def real_examples() -> tuple[np.ndarray, np.ndarray]:
    """Coordonnées des captures réelles (chiffres à gauche, lettres en bas) et coins vides."""
    feats, labels = [], []
    for capture in [START, *chesscom_captures(), PHONE]:
        cells, _ = capture_cells(capture)
        cells = cells.reshape(8, 8, CELL, CELL, 3)
        wb = capture.white_bottom
        digits = [
            GLYPHS.index(str(chess.square_rank(screen_to_square(r, 0, wb)) + 1)) for r in range(8)
        ]
        letters = [
            GLYPHS.index(chess.FILE_NAMES[chess.square_file(screen_to_square(7, c, wb))])
            for c in range(8)
        ]
        feats += [
            corner_features(cells[:, 0], DIGIT_ZONE),
            corner_features(cells[7, :], LETTER_ZONE),
        ]
        labels += digits + letters
        # Coins sans coordonnée : cases du centre.
        feats += [
            corner_features(cells[2:6, 3], DIGIT_ZONE),
            corner_features(cells[2:6, 4], LETTER_ZONE),
        ]
        labels += [0] * 8
    return np.concatenate(feats), np.array(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exemples", type=int, default=30_000)
    parser.add_argument("--epoques", type=int, default=8)
    parser.add_argument("--sortie", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    rng = np.random.default_rng(1)
    start = time.perf_counter()
    fonts = latin_fonts()
    sets = default_sets(rng)
    print(f"{len(fonts)} polices", flush=True)

    feats = np.empty((args.exemples, 32, 32, 6), np.float32)
    labels = np.empty(args.exemples, np.int64)
    for i in range(args.exemples):
        cell, zone, label = synth(rng, fonts, sets)
        feats[i] = corner_features(cell[None], DIGIT_ZONE if zone == 0 else LETTER_ZONE)[0]
        labels[i] = label
    real_x, real_y = real_examples()
    print(
        f"{args.exemples} exemples synthétiques en {time.perf_counter() - start:.0f} s", flush=True
    )

    net = TileNet.initialize(seed=1, classes=len(GLYPHS))
    optimizer = Adam(net.params, lr=2e-3)
    batch = 128
    steps = args.epoques * (len(labels) // batch)
    step = 0
    for epoch in range(1, args.epoques + 1):
        order = rng.permutation(len(labels))
        losses = []
        for i in range(0, len(order) - batch + 1, batch):
            index = order[i : i + batch]
            loss, grads, _ = net.loss_and_gradients(feats[index], labels[index])
            optimizer.lr = 2e-3 * 0.5 * (1 + np.cos(np.pi * step / steps))
            optimizer.step(net.params, grads)
            losses.append(loss)
            step += 1
        real = (net.predict(real_x).argmax(axis=1) == real_y).mean()
        print(
            f"époque {epoch} : perte {np.mean(losses):.3f}, coordonnées réelles {real:.2%} "
            f"({time.perf_counter() - start:.0f} s)",
            flush=True,
        )
    net.save(args.sortie)
    predicted = net.predict(real_x).argmax(axis=1)
    wrong = np.flatnonzero(predicted != real_y)
    print(f"\nModèle enregistré : {args.sortie}")
    print(
        f"Coordonnées réelles : {len(real_y) - len(wrong)}/{len(real_y)} justes ; erreurs : "
        + ", ".join(f"{GLYPHS[predicted[i]]} au lieu de {GLYPHS[real_y[i]]}" for i in wrong[:15])
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

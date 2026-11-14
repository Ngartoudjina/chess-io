"""Entraînement du réseau des cases (lot L2) et mesure sur de vraies captures.

Données : cases synthétiques (``chessio.vision.dataset``) dont les pièces viennent
de polices, du jeu SVG de python-chess et de pièces réelles découpées dans des
captures étiquetées. Validation : les 20 captures Chess.com de la recette L1
(jamais vues à l'entraînement) et la capture de téléphone.

    python scripts/entrainer_cases.py --sans-telephone   # mesure de généralisation
    python scripts/entrainer_cases.py                    # modèle final
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import chess  # noqa: E402
import numpy as np  # noqa: E402

from chessio.vision.cnn import LABELS, MODEL_PATH, Adam, TileNet, features, features_from_reduced  # noqa: E402
from chessio.vision.dataset import LabeledCapture, capture_cells, default_sets, generate, real_set  # noqa: E402
from chessio.vision.ops import background  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "img"
PHONE = LabeledCapture(
    IMG / "telephone_01.png", "r1b1k2r/5ppp/pqn1pn2/1p6/3pP1P1/P1N1BPbP/1PP1K3/R2Q1BNR", True
)
START = LabeledCapture(IMG / "00.png", chess.STARTING_BOARD_FEN, True)


def chesscom_captures() -> list[LabeledCapture]:
    """Les 20 captures de la recette L1, avec la position qu'elles montrent."""
    spec = importlib.util.spec_from_file_location(
        "evaluer", ROOT / "scripts" / "evaluer_captures.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    captures = []
    for sample in module.SAMPLES:
        if sample.file is not None:  # captures hors Chess.com (téléphone) : à part
            continue
        board = sample.expected
        if sample.filename == "10.png":  # coup non joué : la capture montre la position de départ
            board = chess.Board(sample.start_fen)
        captures.append(
            LabeledCapture(IMG / sample.filename, board.board_fen(), not sample.black_bottom)
        )
    return captures


def labeled(captures: list[LabeledCapture]) -> tuple[np.ndarray, np.ndarray]:
    cells, labels = [], []
    for capture in captures:
        capture_cell, capture_labels = capture_cells(capture)
        cells.append(capture_cell)
        labels += [LABELS.index(x) for x in capture_labels]
    return np.concatenate(cells), np.array(labels)


def reduce_cells(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cases 64 × 64 → (32 × 32 uint8, fond) : quatre fois moins de mémoire."""
    reduced = cells.reshape(-1, 32, 2, 32, 2, 3).mean(axis=(2, 4)).round().astype(np.uint8)
    return reduced, background(cells).astype(np.float32)


def accuracy(net: TileNet, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    probs = net.predict(x)
    return float((probs.argmax(axis=1) == y).mean()), probs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sans-telephone", action="store_true", help="exclure les pièces du téléphone"
    )
    parser.add_argument("--cases", type=int, default=60_000, help="cases synthétiques")
    parser.add_argument("--epoques", type=int, default=10)
    parser.add_argument("--lot", type=int, default=128)
    parser.add_argument("--graine", type=int, default=0)
    parser.add_argument("--sortie", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    rng = np.random.default_rng(args.graine)
    start = time.perf_counter()

    sets = default_sets(rng)
    sets.append(real_set("chesscom-neo", [START]))
    if not args.sans_telephone:
        sets.append(real_set("telephone", [PHONE]))
    print("jeux de pièces :", ", ".join(s.name for s in sets), flush=True)

    cells, y = generate(rng, sets, args.cases)
    reduced, bg = reduce_cells(cells)
    del cells
    print(f"{len(y)} cases synthétiques en {time.perf_counter() - start:.0f} s", flush=True)

    checks = {
        "Chess.com (20 captures)": labeled(chesscom_captures()),
        "téléphone (1 capture)": labeled([PHONE]),
    }
    checks = {name: (features(c), labels) for name, (c, labels) in checks.items()}

    net = TileNet.initialize(seed=args.graine)
    optimizer = Adam(net.params, lr=2e-3)
    steps = args.epoques * (len(y) // args.lot)
    step = 0
    for epoch in range(1, args.epoques + 1):
        order = rng.permutation(len(y))
        losses, hits = [], 0
        for i in range(0, len(order) - args.lot + 1, args.lot):
            batch = order[i : i + args.lot]
            x = features_from_reduced(reduced[batch] / 255, bg[batch] / 255)
            loss, grads, probs = net.loss_and_gradients(x, y[batch])
            optimizer.lr = 2e-3 * 0.5 * (1 + np.cos(np.pi * step / steps))  # décroissance cosinus
            optimizer.step(net.params, grads)
            losses.append(loss)
            hits += int((probs.argmax(axis=1) == y[batch]).sum())
            step += 1
        report = [
            f"époque {epoch:2d} : perte {np.mean(losses):.3f}, synthétique {hits / len(order):.2%}"
        ]
        for name, (x, labels) in checks.items():
            acc, _ = accuracy(net, x, labels)
            report.append(f"{name} {acc:.2%}")
        print(" · ".join(report) + f" ({time.perf_counter() - start:.0f} s)", flush=True)

    net.save(args.sortie)
    print(f"\nModèle enregistré : {args.sortie}")
    for name, (x, labels) in checks.items():
        acc, probs = accuracy(net, x, labels)
        wrong = np.flatnonzero(probs.argmax(axis=1) != labels)
        detail = ", ".join(
            f"case {i % 64} de la capture {i // 64 + 1} : {LABELS[probs[i].argmax()]} au lieu de "
            f"{LABELS[labels[i]]} ({probs[i].max():.0%})"
            for i in wrong[:10]
        )
        right_confidence = probs.max(axis=1)[probs.argmax(axis=1) == labels]
        print(
            f"{name} : {len(labels) - len(wrong)}/{len(labels)} cases justes ({acc:.2%}), "
            f"confiance minimale des cases justes {right_confidence.min():.0%}"
            + (f"\n  erreurs : {detail}" if detail else "")
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

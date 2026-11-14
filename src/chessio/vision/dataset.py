"""Données d'entraînement du réseau des cases (lot L2, cahier des charges section 8).

On fabrique des cases normalisées (64 × 64, comme après ``crop_board``) en
combinant au hasard :

* un **jeu de pièces** : glyphes de polices (deux styles), pièces SVG de
  python-chess, ou pièces **réelles** découpées dans des captures étiquetées
  (avec leur opacité estimée, voir :func:`chessio.vision.templates.silhouette`) ;
* des **couleurs de plateau** : thèmes connus et couleurs tirées au hasard ;
* les **perturbations du monde réel** : surlignage du dernier coup, coordonnées
  dans les coins, flèches, barre de défilement, grille légèrement décalée,
  changement de résolution, compression JPEG, bruit.

La couleur de fond des cases change à chaque tirage : le réseau apprend la forme
des pièces, pas les couleurs d'un thème.
"""

from __future__ import annotations

import colorsys
import io
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import chess
import chess.svg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chessio.vision.board import CELL, crop_board, detect_board, screen_to_square, split_cells
from chessio.vision.cnn import LABELS
from chessio.vision.ops import background
from chessio.vision.templates import silhouette

FONT_DIRS = (Path("/usr/share/fonts/truetype"), Path("/usr/share/fonts"))
PIECE_FONTS = ("dejavu/DejaVuSans.ttf", "noto/NotoSansSymbols2-Regular.ttf")
TEXT_FONTS = (
    "dejavu/DejaVuSans.ttf",
    "dejavu/DejaVuSans-Bold.ttf",
    "dejavu/DejaVuSerif.ttf",
    "noto/NotoSans-Regular.ttf",
    "noto/NotoSans-Bold.ttf",
    "ubuntu/Ubuntu-R.ttf",
    "ubuntu/Ubuntu-B.ttf",
)
FILLED = {"k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}
OUTLINE = {"k": "♔", "q": "♕", "r": "♖", "b": "♗", "n": "♘", "p": "♙"}

# Thèmes courants (case claire, case foncée) : Chess.com, Lichess et autres.
THEMES = (
    ((235, 236, 208), (115, 149, 82)),  # Chess.com vert
    ((240, 217, 181), (181, 136, 99)),  # brun (Lichess, Chess.com)
    ((222, 227, 230), (140, 162, 173)),  # Lichess bleu
    ((234, 233, 210), (75, 115, 153)),  # Chess.com bleu
    ((239, 239, 239), (136, 119, 183)),  # violet
    ((220, 220, 220), (171, 171, 171)),  # gris
    ((255, 255, 221), (134, 166, 102)),  # Lichess vert
    ((237, 214, 176), (184, 135, 98)),  # beige
    ((240, 216, 191), (186, 85, 70)),  # rouge
    ((252, 229, 193), (208, 139, 24)),  # orange
    ((205, 217, 228), (81, 109, 143)),  # océan
    ((236, 236, 236), (193, 193, 193)),  # clair
    ((232, 235, 239), (125, 135, 150)),  # ardoise
    ((245, 245, 220), (119, 149, 86)),  # téléphone (vert)
)
# Surlignages : (couleur, opacité) du dernier coup, de la sélection, des prémouvements.
TINTS = (
    ((255, 255, 51), 0.5),
    ((155, 199, 0), 0.41),
    ((20, 85, 30), 0.5),
    ((235, 97, 80), 0.6),
    ((86, 161, 220), 0.5),
    ((150, 100, 60), 0.85),  # case du roi maté (Chess.com)
)


def font_path(relative: str) -> Path | None:
    for base in FONT_DIRS:
        path = base / relative
        if path.exists():
            return path
    return None


@lru_cache(maxsize=64)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@dataclass
class PieceSet:
    """Jeu de pièces : des images RGBA 64 × 64 par symbole FEN."""

    name: str
    sprites: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def add(self, symbol: str, rgba: np.ndarray) -> None:
        self.sprites.setdefault(symbol, []).append(rgba)

    def complete(self) -> bool:
        return all(symbol in self.sprites for symbol in LABELS[1:-1])


# --- jeux de pièces ------------------------------------------------------------


def glyph_set(path: Path, style: str, rng: np.random.Generator, variants: int = 3) -> PieceSet:
    """Pièces dessinées avec les glyphes d'échecs d'une police.

    ``plein`` : glyphe plein, blanc cerné de sombre ou noir cerné de clair ;
    ``classique`` : pièce blanche = contour noir sur fond blanc, pièce noire pleine.
    """
    pieces = PieceSet(f"{path.stem}-{style}")
    for symbol in LABELS[1:-1]:
        lower = symbol.lower()
        white = symbol.isupper()
        for _ in range(variants):
            size = int(rng.integers(44, 56))
            font = _font(str(path), size)
            image = Image.new("RGBA", (CELL, CELL), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            center = (CELL / 2, CELL / 2 + 1)
            if style == "plein":
                if white:
                    fill = tuple(int(v) for v in rng.integers(232, 256, 3)) + (255,)
                    stroke = tuple(int(v) for v in rng.integers(15, 70, 3)) + (255,)
                else:
                    fill = tuple(int(v) for v in rng.integers(15, 75, 3)) + (255,)
                    stroke = tuple(int(v) for v in rng.integers(120, 235, 3)) + (255,)
                draw.text(
                    center, FILLED[lower], font=font, fill=fill, anchor="mm",
                    stroke_width=int(rng.integers(1, 3)), stroke_fill=stroke,
                )  # fmt: skip
            else:
                ink = tuple(int(v) for v in rng.integers(0, 50, 3)) + (255,)
                if white:
                    paper = tuple(int(v) for v in rng.integers(235, 256, 3)) + (255,)
                    draw.text(center, FILLED[lower], font=font, fill=paper, anchor="mm")
                    draw.text(center, OUTLINE[lower], font=font, fill=ink, anchor="mm")
                else:
                    draw.text(center, FILLED[lower], font=font, fill=ink, anchor="mm")
            pieces.add(symbol, np.asarray(image, dtype=np.uint8).copy())
    return pieces


def svg_set() -> PieceSet:
    """Pièces SVG de python-chess (jeu « cburnett »), rendues par Qt."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QByteArray, Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    _app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841 (requis par Qt)
    pieces = PieceSet("svg-cburnett")
    for symbol in LABELS[1:-1]:
        svg = chess.svg.piece(chess.Piece.from_symbol(symbol))
        image = QImage(CELL, CELL, QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        QSvgRenderer(QByteArray(svg.encode())).render(painter)
        painter.end()
        buffer = np.frombuffer(image.constBits(), np.uint8, count=CELL * CELL * 4)
        pieces.add(symbol, buffer.reshape(CELL, CELL, 4).copy())
    return pieces


def sprite_from_cell(cell: np.ndarray) -> np.ndarray | None:
    """Pièce réelle découpée dans une case : couleur de la pièce et opacité estimée."""
    alpha = silhouette(cell)
    if alpha.mean() < 0.05:
        return None
    cell = cell.astype(np.float64)
    bg = background(cell)
    safe = np.maximum(alpha, 1e-3)[..., None]
    color = np.clip((cell - (1 - alpha[..., None]) * bg) / safe, 0, 255)
    rgba = np.dstack([color, alpha * 255]).round().astype(np.uint8)
    return rgba


@dataclass(frozen=True)
class LabeledCapture:
    """Capture réelle dont la position est connue (étiquettes gratuites, section 8)."""

    path: Path
    placement: str  # partie « placement » de la FEN
    white_bottom: bool


def capture_cells(capture: LabeledCapture) -> tuple[np.ndarray, list[str]]:
    """Les 64 cases (ordre de l'écran) et leurs étiquettes."""
    from chessio.vision.board import load_image

    image = load_image(capture.path)
    region, _ = detect_board(image)
    cells = split_cells(crop_board(image, region)).reshape(64, CELL, CELL, 3)
    board = chess.Board(None)
    board.set_board_fen(capture.placement)
    labels = []
    for row in range(8):
        for col in range(8):
            piece = board.piece_at(screen_to_square(row, col, capture.white_bottom))
            labels.append(piece.symbol() if piece else ".")
    return cells, labels


def real_set(name: str, captures: list[LabeledCapture]) -> PieceSet:
    pieces = PieceSet(name)
    for capture in captures:
        cells, labels = capture_cells(capture)
        for cell, label in zip(cells, labels, strict=True):
            if label != ".":
                sprite = sprite_from_cell(cell)
                if sprite is not None:
                    pieces.add(label, sprite)
    return pieces


# --- composition d'une case -----------------------------------------------------


def random_theme(rng: np.random.Generator) -> tuple[tuple, tuple]:
    if rng.random() < 0.6:
        light, dark = THEMES[int(rng.integers(len(THEMES)))]
    else:
        h = rng.random()
        light = colorsys.hsv_to_rgb(h, rng.uniform(0, 0.35), rng.uniform(0.8, 1.0))
        dark = colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.15, 0.7), rng.uniform(0.3, 0.7))
        light = tuple(int(255 * v) for v in light)
        dark = tuple(int(255 * v) for v in dark)
    jitter = lambda c: tuple(int(np.clip(v + rng.integers(-6, 7), 0, 255)) for v in c)  # noqa: E731
    return jitter(light), jitter(dark)


def _blend(base: np.ndarray, color: tuple, alpha: float) -> np.ndarray:
    return base * (1 - alpha) + np.asarray(color, np.float64) * alpha


def compose_cell(
    rng: np.random.Generator, label: str, sets: list[PieceSet], text_fonts: list[Path]
) -> np.ndarray:
    """Une case 64 × 64 (uint8) portant ``label`` (« . » : vide), au hasard."""
    margin = 8
    side = CELL + 2 * margin
    light, dark = random_theme(rng)
    is_light = rng.random() < 0.5
    own, other = (light, dark) if is_light else (dark, light)
    canvas = np.empty((side, side, 3), np.float64)
    canvas[:] = other  # les cases voisines, visibles si la grille est décalée
    canvas[margin : margin + CELL, margin : margin + CELL] = own
    cell_area = (slice(margin, margin + CELL), slice(margin, margin + CELL))
    if rng.random() < 0.12:
        color, alpha = TINTS[int(rng.integers(len(TINTS)))]
        canvas[cell_area] = _blend(canvas[cell_area], color, alpha)
    if label.lower() == "k" and rng.random() < 0.08:  # roi en échec : halo rouge
        yy, xx = np.mgrid[0:CELL, 0:CELL]
        radius = np.hypot(yy - CELL / 2, xx - CELL / 2) / (CELL / 2)
        glow = np.clip(1 - radius, 0, 1)[..., None] * 0.8
        canvas[cell_area] = canvas[cell_area] * (1 - glow) + np.array((230, 30, 30)) * glow

    image = Image.fromarray(canvas.round().astype(np.uint8))
    if rng.random() < 0.3:  # coordonnées dans les coins
        draw = ImageDraw.Draw(image)
        font = _font(str(text_fonts[int(rng.integers(len(text_fonts)))]), int(rng.integers(9, 16)))
        ink = other if rng.random() < 0.7 else tuple(int(v) for v in rng.integers(90, 200, 3))
        if rng.random() < 0.6:
            draw.text((margin + 3, margin + 2), str(rng.integers(1, 9)), font=font, fill=ink)
        else:
            text = "abcdefgh"[int(rng.integers(8))]
            draw.text(
                (margin + CELL - 4, margin + CELL - 3), text, font=font, fill=ink, anchor="rb"
            )

    if label not in (".", "?"):
        pieces = [s for s in sets if label in s.sprites]
        sprites = pieces[int(rng.integers(len(pieces)))].sprites[label]
        sprite = Image.fromarray(sprites[int(rng.integers(len(sprites)))])
        scale = rng.uniform(0.82, 1.08)
        size = max(8, int(round(CELL * scale)))
        sprite = sprite.resize((size, size), Image.Resampling.LANCZOS)
        offset = (CELL - size) // 2
        x = margin + offset + int(rng.integers(-3, 4))
        y = margin + offset + int(rng.integers(-3, 4))
        image.paste(sprite, (x, y), sprite)

    draw = ImageDraw.Draw(image, "RGBA")
    if label == "?":  # objet étranger posé sur la case
        for _ in range(int(rng.integers(1, 3))):
            size = rng.uniform(0.25, 0.7) * CELL
            x0 = margin + rng.uniform(0, CELL - size)
            y0 = margin + rng.uniform(0, CELL - size)
            fill = tuple(int(v) for v in rng.integers(0, 256, 3)) + (int(rng.integers(200, 256)),)
            box = (x0, y0, x0 + size, y0 + size * rng.uniform(0.5, 1.5))
            [draw.rectangle, draw.ellipse][int(rng.integers(2))](box, fill=fill)
    if label.lower() == "k" and rng.random() < 0.15:  # fin de partie : pastille sur le roi
        radius = int(rng.integers(8, 13))
        cx, cy = margin + CELL - radius + 2, margin + radius - 2
        color = ((220, 40, 40), (120, 170, 60), (90, 90, 90))[int(rng.integers(3))]
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color + (255,))
        draw.text((cx, cy), "#", fill=(20, 20, 20, 255), anchor="mm")
    if rng.random() < 0.05:  # flèche d'analyse
        color = ((255, 170, 0), (21, 120, 27), (0, 48, 136))[int(rng.integers(3))]
        angle = rng.uniform(0, np.pi)
        cx, cy = side / 2 + rng.uniform(-10, 10), side / 2 + rng.uniform(-10, 10)
        dx, dy = np.cos(angle) * side, np.sin(angle) * side
        draw.line(
            (cx - dx, cy - dy, cx + dx, cy + dy), fill=color + (int(rng.integers(110, 200)),),
            width=int(rng.integers(6, 13)),
        )  # fmt: skip
    if rng.random() < 0.03:  # barre de défilement d'un téléphone
        x0 = margin + int(rng.integers(0, 8))
        draw.rounded_rectangle(
            (x0, margin, x0 + int(rng.integers(5, 10)), margin + CELL),
            radius=4, fill=(60, 60, 60, int(rng.integers(90, 180))),
        )  # fmt: skip

    dx, dy = (int(v) for v in rng.integers(-3, 4, 2))  # grille légèrement décalée
    image = image.crop((margin + dx, margin + dy, margin + dx + CELL, margin + dy + CELL))
    return degrade(rng, np.asarray(image, dtype=np.uint8))


def degrade(rng: np.random.Generator, cell: np.ndarray) -> np.ndarray:
    """Résolution d'origine, compression, bruit et contraste variables."""
    image = Image.fromarray(cell)
    factor = rng.uniform(0.35, 1.3)
    if abs(factor - 1) > 0.05:
        size = max(10, int(round(CELL * factor)))
        image = image.resize((size, size), Image.Resampling.BOX if factor < 1 else Image.BILINEAR)
        image = image.resize((CELL, CELL), Image.Resampling.BILINEAR)
    if rng.random() < 0.25:
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=int(rng.integers(35, 92)))
        image = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    out = np.asarray(image, dtype=np.float64)
    if rng.random() < 0.3:
        out = out * rng.uniform(0.9, 1.1) + rng.uniform(-10, 10)
    if rng.random() < 0.3:
        out = out + rng.normal(0, rng.uniform(0.5, 3), out.shape)
    return np.clip(out, 0, 255).round().astype(np.uint8)


def default_sets(rng: np.random.Generator) -> list[PieceSet]:
    """Jeux synthétiques : glyphes de polices (deux styles) et pièces SVG."""
    sets = []
    for relative in PIECE_FONTS:
        path = font_path(relative)
        if path is not None:
            sets += [glyph_set(path, "plein", rng), glyph_set(path, "classique", rng)]
    sets.append(svg_set())
    return sets


def text_fonts() -> list[Path]:
    fonts = [p for p in (font_path(r) for r in TEXT_FONTS) if p is not None]
    if not fonts:
        raise FileNotFoundError("Aucune police pour les coordonnées (paquet fonts-dejavu-core)")
    return fonts


def generate(
    rng: np.random.Generator, sets: list[PieceSet], count: int, empty_share: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """``count`` cases synthétiques : (cases 64 × 64 × 3 uint8, indices de classe)."""
    fonts = text_fonts()
    draw = rng.random(count)
    labels = np.where(draw < empty_share, 0, rng.integers(1, len(LABELS) - 1, count))
    labels = np.where(draw > 0.95, len(LABELS) - 1, labels).astype(np.int64)  # 5 % d'inconnus
    cells = np.empty((count, CELL, CELL, 3), np.uint8)
    for i, label in enumerate(labels):
        cells[i] = compose_cell(rng, LABELS[label], sets, fonts)
    return cells, labels

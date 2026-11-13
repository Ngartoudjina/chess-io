"""Apparence : palette sombre, feuille de style Qt, icônes et couleurs de l'échiquier."""

from __future__ import annotations

from functools import lru_cache

import chess
import chess.svg
from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

PALETTE = {
    "bg": "#161512",
    "surface": "#1f1d1b",
    "card": "#272522",
    "card_alt": "#312e2b",
    "hover": "#3b3834",
    "border": "#38352f",
    "text": "#f0efec",
    "muted": "#a8a49d",
    "faint": "#77736c",
    "accent": "#81b64c",
    "accent_hover": "#95c95e",
    "accent_text": "#132309",
    "danger": "#e5574b",
    "warning": "#f0b13b",
    "info": "#5aa2e0",
    "white_piece": "#f2f1ee",
    "black_piece": "#121110",
}

# Échiquier aux couleurs du thème vert de Chess.com ; flèches : meilleur coup en
# orange vif, coups suivants en bleu puis violet (couleurs nommées de chess.svg,
# seules à accepter la transparence).
BOARD_COLORS = {
    "square light": "#ebecd0",
    "square dark": "#739552",
    "square light lastmove": "#f5f682",
    "square dark lastmove": "#b9ca43",
    "arrow green": "#f59e0be6",
    "arrow blue": "#38bdf8b3",
    "arrow yellow": "#c084fcb3",
}
ARROW_NAMES = ("green", "blue", "yellow")
DOUBTFUL_FILL = "#e5574b99"
COORD_ON_LIGHT = "#739552"
COORD_ON_DARK = "#ebecd0"


def stylesheet() -> str:
    p = PALETTE
    return f"""
    QWidget {{
        background: {p["bg"]};
        color: {p["text"]};
        font-size: 10pt;
    }}
    QToolTip {{
        background: {p["card_alt"]};
        color: {p["text"]};
        border: 1px solid {p["border"]};
        border-radius: 6px;
        padding: 6px 8px;
    }}
    QFrame#Header {{
        background: {p["surface"]};
        border-bottom: 1px solid {p["border"]};
    }}
    QLabel#AppTitle {{ font-size: 15pt; font-weight: 700; background: transparent; }}
    QLabel#AppSubtitle {{ color: {p["muted"]}; background: transparent; }}
    QFrame#Card {{
        background: {p["card"]};
        border: 1px solid {p["border"]};
        border-radius: 14px;
    }}
    QFrame#Card QLabel, QFrame#Card QWidget#Transparent {{ background: transparent; }}
    QLabel#Caption {{
        color: {p["muted"]};
        font-size: 8pt;
        font-weight: 700;
        letter-spacing: 1px;
    }}
    QLabel#Muted {{ color: {p["muted"]}; }}
    QLabel#Hint {{ color: {p["muted"]}; font-size: 9pt; }}
    QLabel#HintWarning {{ color: {p["warning"]}; font-size: 9pt; }}
    QLabel#BestMove {{ font-size: 30pt; font-weight: 800; }}
    QLabel#EvalChip {{
        background: {p["card_alt"]};
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 13pt;
        font-weight: 700;
    }}
    QLabel#Status {{ color: {p["muted"]}; background: transparent; }}
    QPushButton, QToolButton {{
        background: {p["card_alt"]};
        border: 1px solid {p["border"]};
        border-radius: 9px;
        padding: 7px 14px;
    }}
    QPushButton:hover, QToolButton:hover {{ background: {p["hover"]}; }}
    QPushButton:pressed, QToolButton:pressed {{ background: {p["card"]}; }}
    QPushButton:disabled {{ color: {p["faint"]}; }}
    QPushButton#Primary {{
        background: {p["accent"]};
        color: {p["accent_text"]};
        border: none;
        font-weight: 700;
    }}
    QPushButton#Primary:hover {{ background: {p["accent_hover"]}; }}
    QToolButton#HeaderButton {{
        background: transparent;
        border: none;
        padding: 7px 12px;
    }}
    QToolButton#HeaderButton:hover {{ background: {p["card_alt"]}; }}
    QToolButton::menu-indicator {{ image: none; }}
    QPushButton#Segment {{
        border-radius: 0;
        padding: 8px 10px;
        background: {p["surface"]};
    }}
    QPushButton#Segment[position="first"] {{
        border-top-left-radius: 9px;
        border-bottom-left-radius: 9px;
    }}
    QPushButton#Segment[position="last"] {{
        border-top-right-radius: 9px;
        border-bottom-right-radius: 9px;
    }}
    QPushButton#Segment:checked {{
        background: {p["accent"]};
        color: {p["accent_text"]};
        border-color: {p["accent"]};
        font-weight: 700;
    }}
    QPushButton#TurnWhite, QPushButton#TurnBlack {{
        padding: 12px 10px;
        font-size: 12pt;
        font-weight: 700;
        background: {p["surface"]};
        color: {p["muted"]};
    }}
    QPushButton#TurnWhite {{ border-top-left-radius: 11px; border-bottom-left-radius: 11px; }}
    QPushButton#TurnBlack {{ border-top-right-radius: 11px; border-bottom-right-radius: 11px; }}
    QPushButton#TurnWhite:checked {{
        background: {p["white_piece"]};
        color: #1a1917;
        border: 2px solid {p["accent"]};
    }}
    QPushButton#TurnBlack:checked {{
        background: {p["black_piece"]};
        color: {p["text"]};
        border: 2px solid {p["accent"]};
    }}
    QFrame#LineRow {{
        background: {p["surface"]};
        border: 1px solid {p["border"]};
        border-radius: 10px;
    }}
    QFrame#LineRow:hover {{ background: {p["hover"]}; }}
    QFrame#LineRow[selected="true"] {{ border: 1px solid {p["accent"]}; }}
    QFrame#LineRow QLabel {{ background: transparent; }}
    QFrame#Banner {{
        border-radius: 12px;
        border: 1px solid {p["border"]};
        background: {p["card"]};
    }}
    QFrame#Banner[kind="warning"] {{ border-color: {p["warning"]}; background: #2e2717; }}
    QFrame#Banner[kind="danger"] {{ border-color: {p["danger"]}; background: #2f1b19; }}
    QFrame#Banner[kind="info"] {{ border-color: {p["info"]}; background: #18232e; }}
    QFrame#Banner QLabel {{ background: transparent; }}
    QFrame#DropHint {{
        background: {p["surface"]};
        border: 2px dashed {p["border"]};
        border-radius: 18px;
    }}
    QFrame#DropHint QLabel {{ background: transparent; }}
    QLabel#DropTitle {{ font-size: 17pt; font-weight: 700; }}
    QLabel#Thumbnail {{
        background: {p["surface"]};
        border: 1px solid {p["border"]};
        border-radius: 10px;
    }}
    QLineEdit {{
        background: {p["surface"]};
        border: 1px solid {p["border"]};
        border-radius: 9px;
        padding: 7px 10px;
        font-family: monospace;
        selection-background-color: {p["accent"]};
    }}
    QLineEdit:focus {{ border-color: {p["accent"]}; }}
    QSlider::groove:horizontal {{
        height: 6px;
        background: {p["card_alt"]};
        border-radius: 3px;
    }}
    QSlider::sub-page:horizontal {{ background: {p["accent"]}; border-radius: 3px; }}
    QSlider::handle:horizontal {{
        width: 16px;
        margin: -6px 0;
        border-radius: 8px;
        background: {p["text"]};
    }}
    QProgressBar {{
        background: {p["card_alt"]};
        border: none;
        border-radius: 2px;
        max-height: 4px;
    }}
    QProgressBar::chunk {{ background: {p["accent"]}; border-radius: 2px; }}
    QMenu {{
        background: {p["card"]};
        border: 1px solid {p["border"]};
        border-radius: 10px;
        padding: 6px;
    }}
    QMenu::item {{ padding: 7px 22px 7px 12px; border-radius: 7px; }}
    QMenu::item:selected {{ background: {p["hover"]}; }}
    QMenu::separator {{ height: 1px; background: {p["border"]}; margin: 5px 8px; }}
    QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{
        background: {p["card_alt"]};
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QLabel#Toast {{
        background: {p["card_alt"]};
        color: {p["text"]};
        border: 1px solid {p["border"]};
        border-radius: 10px;
        padding: 9px 16px;
    }}
    QDialog {{ background: {p["surface"]}; }}
    QMessageBox QLabel {{ background: transparent; }}
    """


# Icônes au trait (style Feather), dessinées en SVG pour rester nettes à toute taille.
_ICONS = {
    "paste": '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>'
    '<rect x="8" y="2" width="8" height="4" rx="1"/>',
    "open": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "copy": '<rect x="9" y="9" width="12" height="12" rx="2"/>'
    '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
    "flip": '<path d="M7 20V4M3 8l4-4 4 4M17 4v16M13 16l4 4 4-4"/>',
    "crop": '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/>'
    '<circle cx="12" cy="12" r="1"/>',
    "menu": '<circle cx="12" cy="5" r="1.5"/><circle cx="12" cy="12" r="1.5"/>'
    '<circle cx="12" cy="19" r="1.5"/>',
    "deeper": '<path d="M12 4v14M6 12l6 6 6-6"/><path d="M5 21h14"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/>'
    '<path d="M21 15l-5-5L5 21"/>',
    "check": '<path d="M20 6L9 17l-5-5"/>',
    "alert": '<path d="M12 9v4M12 17h.01"/>'
    '<path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    "play": '<path d="M7 4l12 8-12 8z"/>',
    "close": '<path d="M18 6L6 18M6 6l12 12"/>',
}


@lru_cache(maxsize=128)
def icon(name: str, color: str = PALETTE["text"], size: int = 20) -> QIcon:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f"{_ICONS[name]}</svg>"
    )
    return QIcon(svg_pixmap(svg, size))


def svg_pixmap(svg: str, size: int, scale: int = 2) -> QPixmap:
    """Rend un SVG en image nette (rendu à ``scale``× pour les écrans haute densité)."""
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(svg.encode())).render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(scale)
    return pixmap


@lru_cache(maxsize=16)
def piece_icon(symbol: str, size: int = 28) -> QIcon:
    return QIcon(svg_pixmap(chess.svg.piece(chess.Piece.from_symbol(symbol)), size))


def app_icon() -> QIcon:
    return QIcon(svg_pixmap(chess.svg.piece(chess.Piece(chess.KNIGHT, chess.WHITE)), 64))

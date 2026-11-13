"""Couleurs de cases d'un thème de plateau."""

from __future__ import annotations

from dataclasses import dataclass

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    """Couleurs des cases claires et foncées, y compris surlignées (dernier coup)."""

    name: str
    light: tuple[RGB, ...]
    dark: tuple[RGB, ...]

    def to_dict(self) -> dict:
        return {"name": self.name, "light": list(self.light), "dark": list(self.dark)}

    @classmethod
    def from_dict(cls, data: dict) -> Theme:
        return cls(
            name=str(data["name"]),
            light=tuple(tuple(int(v) for v in c) for c in data["light"]),
            dark=tuple(tuple(int(v) for v in c) for c in data["dark"]),
        )


# Thème « Vert » de Chess.com (ancienne et nouvelle teinte), cases normales puis
# surlignées par le dernier coup. Remplacé par les couleurs mesurées lors de la
# calibration.
CHESSCOM_GREEN = Theme(
    name="chess.com vert",
    light=((235, 236, 208), (238, 238, 210), (245, 246, 130)),
    dark=((115, 149, 82), (118, 150, 86), (185, 202, 67)),
)

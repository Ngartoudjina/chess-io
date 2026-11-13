"""Couleurs de cases d'un thème de plateau."""

from __future__ import annotations

from dataclasses import dataclass

RGB = tuple[int, int, int]

# Surlignage du dernier coup sur Chess.com (jaune à 50 % sur le thème vert),
# mesuré sur des captures réelles : case claire puis case foncée.
CHESSCOM_HIGHLIGHT: tuple[RGB, ...] = ((245, 246, 129), (185, 202, 66))


@dataclass(frozen=True)
class Theme:
    """Couleurs des cases claires et foncées (y compris surlignées) et du surlignage."""

    name: str
    light: tuple[RGB, ...]
    dark: tuple[RGB, ...]
    highlight: tuple[RGB, ...] = CHESSCOM_HIGHLIGHT

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "light": list(self.light),
            "dark": list(self.dark),
            "highlight": list(self.highlight),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Theme:
        def colors(key: str) -> tuple[RGB, ...]:
            return tuple(tuple(int(v) for v in c) for c in data[key])

        return cls(
            name=str(data["name"]),
            light=colors("light"),
            dark=colors("dark"),
            # Les calibrations antérieures n'enregistraient pas le surlignage.
            highlight=colors("highlight") if "highlight" in data else CHESSCOM_HIGHLIGHT,
        )


# Thème « Vert » de Chess.com (ancienne et nouvelle teinte), cases normales puis
# surlignées par le dernier coup. Remplacé par les couleurs mesurées lors de la
# calibration.
CHESSCOM_GREEN = Theme(
    name="chess.com vert",
    light=((235, 236, 208), (238, 238, 210), (245, 246, 130)),
    dark=((115, 149, 82), (118, 150, 86), (185, 202, 67)),
)

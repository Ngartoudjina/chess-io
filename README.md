# Chessio

Assistant « meilleur coup » hors-ligne pour Ubuntu : à partir d'une capture
d'écran d'un échiquier Chess.com et de la couleur au trait, il reconstitue la
position (FEN) et affiche le meilleur coup.

Le cahier des charges complet est dans
[Cahier_des_charges_meilleur_coup.pdf](Cahier_des_charges_meilleur_coup.pdf).

> **Cadre d'utilisation.** Chess.com interdit toute aide d'un moteur pendant une
> partie en cours. Chessio est destiné à l'analyse de parties terminées, aux
> parties contre les bots, aux problèmes et à l'étude, et ne joue jamais à la
> place de l'utilisateur.

## État d'avancement

| Lot | Contenu | État |
|-----|---------|------|
| L0 — Socle | Dépôt, environnement, FEN → Stockfish, CI | ✅ en cours de recette |
| L1 — Chaîne minimale | Capture → coup, un thème | à faire |
| L2 — Vision robuste | Classifieur ONNX, orientation, correction | à faire |
| L3 — Moteur maison v1 | Python, négamax alpha-bêta, UCI | à faire |
| L4 — Moteur maison v2 | Rust + shakmaty, réglage Texel | à faire |
| L5 — Évaluation apprise | Petit réseau NNUE | à faire |
| L6 — Finition | AppImage, documentation, recette | à faire |

## Installation (développement)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
scripts/fetch_stockfish.sh      # Stockfish 19, binaire « universal », dans engines/
```

Un autre Stockfish peut être utilisé via `CHESSIO_STOCKFISH=/chemin/vers/stockfish`
ou s'il est présent dans le `PATH`.

## Utilisation (lot L0 : ligne de commande)

```bash
# FEN complète
.venv/bin/chessio analyse "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"

# Placement seul + couleur au trait (les droits de roque sont déduits)
.venv/bin/chessio analyse rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR --trait b --multipv 3 --temps 2
```

Une position invalide (deux rois, pion en 1re/8e rangée, camp sans le trait en
échec…) est refusée avec la liste des cases en cause : aucun coup n'est proposé.

## Organisation

```
src/chessio/
  position.py   construction de la FEN (roque déduit) et validation (EF-06, EF-07)
  engine.py     moteur UCI dans un processus séparé, temps 0,1–10 s, MultiPV (EF-08, EF-10)
  __main__.py   ligne de commande
tests/          pytest (règles, moteur, CLI)
scripts/        téléchargement de Stockfish
```

## Tests

```bash
.venv/bin/ruff check .
.venv/bin/pytest
```

## Licence

GPL 3 ou ultérieure (python-chess, shakmaty et Stockfish sont sous GPL 3).

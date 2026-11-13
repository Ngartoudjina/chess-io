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
| L0 — Socle | Dépôt, environnement, FEN → Stockfish, CI | ✅ terminé |
| L1 — Chaîne minimale | Capture → coup, un thème, orientation automatique | ✅ recette passée : 20/20 captures réelles |
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

## Utilisation

### Première fois : calibration

Faites une capture d'écran de la **position initiale** dans votre thème Chess.com
habituel, puis :

```bash
.venv/bin/chessio calibrer position_initiale.png
```

Les gabarits des 12 pièces sont enregistrés dans `~/.local/share/chessio/`.
(Dans l'interface : Fichier → Calibrer.)

### Interface graphique

```bash
.venv/bin/chessio gui
```

Collez une capture (Ctrl+V), glissez-la ou ouvrez-la (Ctrl+O), indiquez le trait
(touches W / B) : la position est reconnue puis analysée, le meilleur coup est
fléché sur le mini-échiquier. Les cases douteuses sont marquées en rouge et aucun
coup n'est proposé tant qu'elles ne sont pas corrigées (champ FEN modifiable).

L'orientation (Blancs ou Noirs en bas) est lue sur les **coordonnées** que
Chess.com écrit sur les cases : gardez le réglage « Coordonnées : à l'intérieur
du plateau ». Sans elles, l'orientation est déduite de la structure de pions ; si
elle reste incertaine, aucun coup n'est proposé avant que vous la précisiez.

### Ligne de commande

```bash
# Capture d'écran → meilleur coup
.venv/bin/chessio capture partie.png --trait w

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
  vision/board.py      localisation du plateau, découpe en 64 cases (EF-02)
  vision/templates.py  calibration et classification par gabarits (EF-05)
  vision/orientation.py  orientation par les coordonnées affichées (EF-04)
  vision/ops.py        opérations d'image communes (NumPy)
  vision/synthetic.py  captures synthétiques pour les tests
  recognize.py         capture + trait → position validée (seule jonction vision/moteur)
  position.py          construction de la FEN (roque déduit) et validation (EF-06, EF-07)
  engine.py            moteur UCI dans un processus séparé, temps 0,1–10 s, MultiPV (EF-08, EF-10)
  gui.py               interface PySide6 (EF-01, EF-03, EF-09)
  __main__.py          ligne de commande
tests/                 pytest (règles, vision, moteur, CLI)
scripts/               téléchargement de Stockfish, recette sur captures réelles
```

## Tests

```bash
.venv/bin/ruff check .
.venv/bin/pytest
```

Recette sur captures réelles (lot L1) : `scripts/evaluer_captures.py --instructions img/A_CAPTURER.md`
écrit la liste des 20 positions à capturer ; une fois les captures dans `img/` :

```bash
.venv/bin/python scripts/evaluer_captures.py --calibration img/00.png --dossier img
```

Les captures restent locales (`img/` est ignoré par Git) : les graphismes des
pièces de Chess.com ne sont pas libres.

Résultat de la recette L1 (13/11/2026, thème vert, cases de 95 px) : 20/20
positions exactes, 1 280/1 280 cases, orientation 20/20 par les coordonnées
(dont 7 captures où le camp au trait n'est pas en bas), 450 ms au plus.

## Licence

GPL 3 ou ultérieure (python-chess, shakmaty et Stockfish sont sous GPL 3).

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
| L2 — Vision robuste | Réseau des cases, tout thème, sans calibration | ✅ 21/21 captures (dont téléphone) |
| L3 — Moteur maison v1 | Python, négamax alpha-bêta, UCI | ✅ perft exact, Elo mesuré |
| L4 — Moteur maison v2 | Rust + shakmaty, réglage Texel | ✅ ≈ 2 400 Elo (réglage Texel à venir) |
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

### Interface graphique

```bash
.venv/bin/chessio gui
```

**Un seul geste : collez la capture** (Ctrl+V), glissez-la dans la fenêtre ou
ouvrez-la (Ctrl+O). Tout le reste est automatique :

- **Première utilisation** : la première capture doit montrer la **position
  initiale** dans votre thème Chess.com habituel ; Chessio y apprend vos pièces et
  vos couleurs, une fois pour toutes (Menu ⋮ → Recalibrer pour recommencer).
- **Orientation** (Blancs ou Noirs en bas) : lue sur les coordonnées écrites sur
  les cases — gardez le réglage Chess.com « Coordonnées : à l'intérieur du
  plateau ». À défaut, déduite de la structure de pions ; sinon Chessio vous la
  demande.
- **Trait** : déduit du **dernier coup surligné** (ou d'un roi en échec) ; sinon
  choisissez-le (boutons Blancs / Noirs, touches W / B), l'indication passe en
  orange pour vous le rappeler.
- **Prise en passant** : déduite du dernier coup quand un pion vient d'avancer de
  deux cases.
- **Analyse progressive** : le meilleur coup s'affiche aussitôt et s'affine en
  direct ; barre d'évaluation, chances gain / nulle / perte, trois coups au choix
  (cliquez sur une ligne pour voir sa flèche). « Approfondir » (Espace) analyse
  10 s, Échap arrête.
- **Correction en un clic** : une case incertaine est marquée en rouge et aucun
  coup n'est proposé tant qu'elle n'est pas vérifiée ; cliquez sur une case pour
  choisir la bonne pièce.

Raccourcis : Ctrl+V coller, Ctrl+O ouvrir, W / B trait, F retourner l'échiquier,
Espace approfondir, Échap arrêter, Ctrl+Maj+C copier la FEN. La FEN est aussi
modifiable directement sous l'échiquier (Entrée pour analyser).

### Moteur maison (lots L3 et L4)

Le projet a son propre moteur d'échecs, en deux versions de même logique :

- **v0.3 en Rust** (lot L4, `moteur-rs/`) : génération des coups par bitboards
  et tables d'attaque « magiques » (shakmaty), hachage de Zobrist ; environ
  1,3 million de positions par seconde, **≈ 2 400 Elo** ;
- **v0.2 en Python** (lot L3, `chessio/moteur/`) : sert à mettre au point les
  algorithmes, ≈ 1 700 Elo.

Recherche : négamax alpha-bêta, approfondissement itératif avec fenêtres
d'aspiration, recherche à fenêtre nulle (PVS), table de transposition, coup nul,
réductions des coups tardifs (LMR), élagages statique et de frontière,
quiescence, ordre des coups (MVV-LVA, coups tueurs, historique). Évaluation :
matériel, tables pièce-case interpolées milieu / finale, pions passés, doublés et
isolés, tours sur colonne ouverte, abri du roi, mobilité.

Les deux parlent UCI comme Stockfish. « maison » (interface : Réglages → Moteur
d'analyse ; ligne de commande : `--moteur maison`) désigne la version Rust si
elle est compilée, sinon la version Python (`--moteur maison-python` la force).

```bash
# Compiler le moteur Rust (Rust installé avec rustup)
cargo build --release --manifest-path moteur-rs/Cargo.toml

.venv/bin/chessio analyse "<FEN>" --moteur maison
printf 'position startpos\ngo perft 5\nquit\n' | moteur-rs/target/release/chessio-moteur
.venv/bin/python -m chessio.moteur            # version Python, en UCI
```

**Match en direct** : `chessio match` ouvre une fenêtre où le moteur maison affronte
Stockfish, bridé au niveau voulu, coup par coup (pause, vitesse réglable). Le score
donne une estimation de son Elo avec un intervalle de confiance à 95 %.

```bash
.venv/bin/chessio match --elo 1800 --parties 20 --temps 0.3          # fenêtre en direct
.venv/bin/chessio match --elo 1800 --sans-fenetre --pgn parties.pgn  # en texte
```

### Jouer contre l'IA

`chessio jouer` (ou menu ⋮ → « Jouer contre l'IA », Ctrl+J) : choisissez vos pièces
et le niveau de l'IA (Stockfish bridé de 1 320 à 2 800 Elo, pleine force, ou le moteur
maison), puis jouez en cliquant la pièce et sa case d'arrivée. Un coach (Stockfish
à pleine force) affiche l'évaluation et donne un indice sur demande (H) ;
Ctrl+Z annule votre dernier coup.

**Assistant** : à chaque tour, un panneau commente la partie en français — qualité
de votre dernier coup (excellent, imprécision, erreur, gaffe) et meilleure
alternative, intention du coup adverse (attaque, fourchette, échec…), **menace**
de l'adversaire (ce qu'il jouerait s'il rejouait), **pièges** (pièces en prise,
prises empoisonnées, mat en vue), conseils de **stratégie** selon la phase de jeu
et la structure, et plan du moteur. Tout est calculé hors-ligne (`chessio/coach.py`).

### Ligne de commande

```bash
# Calibration (une fois) sur une capture de la position initiale
.venv/bin/chessio calibrer position_initiale.png

# Capture d'écran → meilleur coup (trait déduit du dernier coup surligné)
.venv/bin/chessio capture partie.png
.venv/bin/chessio capture partie.png --trait b --multipv 3   # trait imposé

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
  vision/lastmove.py   dernier coup surligné : trait et prise en passant
  vision/ops.py        opérations d'image communes (NumPy)
  vision/synthetic.py  captures synthétiques pour les tests
  recognize.py         capture + trait → position validée (seule jonction vision/moteur)
  position.py          construction de la FEN (roque déduit) et validation (EF-06, EF-07)
  moteur/              moteur maison v1 : évaluation, recherche, perft, protocole UCI (L3)
  engine.py            moteur UCI dans un processus séparé : analyse progressive,
                       interruptible, MultiPV, chances gain/nulle/perte (EF-08, EF-10)
  ui/                  interface PySide6 : fenêtre, composants, style, travail en arrière-plan
                       (EF-01, EF-03, EF-07, EF-09, EF-11)
  __main__.py          ligne de commande
tests/                 pytest (règles, vision, moteur, CLI, interface hors écran)
scripts/               téléchargement de Stockfish, recette sur captures réelles, matchs
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

Résultat de la recette L3 (13/11/2026, moteur maison 0.1) :

| Test | Résultat |
|---|---|
| Perft, position initiale, profondeur 3 | 8 902 nœuds, exact |
| Perft, position initiale, profondeur 5 | 4 865 609 nœuds, exact (14 s) |
| Perft Kiwipete, profondeur 4 | 4 085 603 nœuds, exact |
| Match contre Stockfish bridé à 1 320 Elo (20 parties, 0,3 s/coup) | 18/20 (+18 =0 −2) |
| Match contre Stockfish bridé à 1 800 Elo (20 parties, 0,3 s/coup) | 6,5/20 (+3 =7 −10) |
| **Elo estimé** (40 parties combinées) | **≈ 1 680** (1 550 à 1 820 à 95 %) |

L'échelle est celle du réglage `UCI_Elo` de Stockfish, à 0,3 s par coup : c'est
une estimation de niveau, pas un classement officiel. La vitesse du moteur Python
(~25 000 positions/s, profondeur 5 en une seconde) est sa principale limite ; le
portage en Rust (lot L4) vise surtout à la multiplier.

Résultat de la recette L4 (13/11/2026, moteur Rust 0.3, même protocole) :

| Test | Résultat |
|---|---|
| Perft, position initiale, profondeur 5 | 4 865 609 nœuds, exact (68 ms, 210 × plus rapide qu'en Python) |
| Évaluation identique à la version Python | 200 positions sur 200 |
| Vitesse et profondeur en 1 s | ≈ 1,3 million de positions/s, profondeur 9 à 11 |
| Match contre Stockfish bridé à 2 200 Elo (20 parties) | 14/20 (+13 =2 −5) |
| Match contre Stockfish bridé à 2 500 Elo (20 parties) | 8,5/20 (+7 =3 −10) |
| **Elo estimé** (40 parties combinées) | **≈ 2 400** (2 284 à 2 521 à 95 %) |

Objectif fixé après le lot L3 : plus de 2 200 Elo, atteint. Le réglage Texel de
l'évaluation, prévu au lot L4, reste à faire.

Résultat de la recette L2 (14/11/2026, sans calibration) : 21/21 positions
(20 Chess.com + 1 capture de téléphone d'une autre application), 1 344/1 344 cases,
orientation 21/21 par lecture des coordonnées, 300 ms au plus. Généralisation : le
modèle entraîné sans les pièces du téléphone lisait déjà 63 cases sur 64 de cette
capture. Réseaux en NumPy (`chessio/vision/cnn.py`, ~95 000 poids) plutôt que
PyTorch/ONNX : la connexion rendait PyTorch (200 Mo) impraticable ; réentraînement
avec `scripts/entrainer_cases.py` et `scripts/entrainer_coordonnees.py`.

## Licence

GPL 3 ou ultérieure (python-chess, shakmaty et Stockfish sont sous GPL 3).

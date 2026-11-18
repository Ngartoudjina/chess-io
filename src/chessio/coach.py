"""Assistant d'entraînement : explications en français pendant une partie contre l'IA.

Les textes sont produits par des règles d'échecs appuyées sur les calculs du
moteur (aucun service extérieur). Ils sont hiérarchisés pour rester lisibles :

1. **Votre dernier coup** : bon ou mauvais, et *pourquoi* (la réponse qui le punit) ;
2. **Ce que fait l'IA** : l'idée de son coup et sa **menace** (ce qu'elle jouerait
   si elle rejouait : analyse après un « coup nul ») ;
3. **À faire maintenant** : *une seule* priorité, la plus urgente (mat, pièce en
   prise, coup gagnant, menace à parer, sinon l'idée du meilleur coup) ;
4. **À éviter** : une prise empoisonnée ;
5. **Idée pour la suite** : un conseil de fond, seulement quand rien ne presse.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from chessio.engine import Line
from chessio.ui.notation import figurine

VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
NAMES = {
    chess.PAWN: ("pion", "m"),
    chess.KNIGHT: ("cavalier", "m"),
    chess.BISHOP: ("fou", "m"),
    chess.ROOK: ("tour", "f"),
    chess.QUEEN: ("dame", "f"),
    chess.KING: ("roi", "m"),
}
CENTER = chess.SquareSet([chess.D4, chess.E4, chess.D5, chess.E5])
MATE = 10_000

# Titres affichés, dans l'ordre de lecture.
TITLES = {
    "coup": "Votre dernier coup",
    "adversaire": "Ce que fait l'IA",
    "menace": "Menace",
    "action": "À faire maintenant",
    "éviter": "À éviter",
    "idée": "Idée pour la suite",
    "stratégie": "Idée pour la suite",
}
# Ordre d'affichage : ce qu'il faut faire d'abord, le reste ensuite.
ORDER = {"action": 0, "coup": 1, "adversaire": 2, "menace": 2, "éviter": 3, "idée": 4}


@dataclass(frozen=True)
class Advice:
    kind: str  # clé de TITLES
    text: str
    level: str = "info"  # « bien », « info », « attention », « danger »


# --- outils de rédaction ---------------------------------------------------------------


def piece_name(piece: chess.Piece, article: bool = True) -> str:
    name, gender = NAMES[piece.piece_type]
    color = (
        ("blanc" if gender == "m" else "blanche")
        if piece.color
        else ("noir" if gender == "m" else "noire")
    )
    if not article:
        return f"{name} {color}"
    return f"{'le' if gender == 'm' else 'la'} {name} {color}"


def _your(piece_type: chess.PieceType) -> str:
    """« votre cavalier », « votre dame »."""
    return f"votre {NAMES[piece_type][0]}"


def _mine(board: chess.Board, square: chess.Square) -> str:
    """« votre cavalier en d4 » (pièce du joueur au trait)."""
    piece_type = board.piece_type_at(square)
    return f"{_your(piece_type)} en {chess.square_name(square)}"


def _san(board: chess.Board, move: chess.Move) -> str:
    """Coup en notation figurine (♘f3), lisible quelle que soit la langue."""
    return figurine(board.san(move), board.turn)


def _agree(board: chess.Board, square: chess.Square, masculine: str, feminine: str) -> str:
    piece = board.piece_at(square)
    return feminine if piece is not None and NAMES[piece.piece_type][1] == "f" else masculine


def _at(board: chess.Board, square: chess.Square) -> str:
    piece = board.piece_at(square)
    return (
        f"{piece_name(piece)} en {chess.square_name(square)}"
        if piece
        else chess.square_name(square)
    )


def _pawns(amount: float) -> str:
    """« un demi-pion », « 1 pion », « 3 pions »."""
    if amount < 0.75:
        return "un demi-pion"
    count = round(amount)
    return f"{count} pion{'s' if count > 1 else ''}"


def _cp(line: Line, color: chess.Color) -> int:
    """Évaluation (centipions) du point de vue de ``color`` ; mat ramené à ±10 000."""
    return line.score.pov(color).score(mate_score=MATE)


def _mate(line: Line | None, color: chess.Color) -> int | None:
    """Nombre de coups du mat, positif si ``color`` mate, négatif s'il est maté."""
    return None if line is None else line.score.pov(color).mate()


def _null(board: chess.Board) -> chess.Board:
    """Position où l'adversaire rejouerait (« coup nul »)."""
    null = board.copy(stack=False)
    null.turn = not null.turn
    null.ep_square = None
    return null


# --- analyse de la position --------------------------------------------------------------


def hanging(board: chess.Board, color: chess.Color) -> list[chess.Square]:
    """Pièces de ``color`` attaquées et insuffisamment défendues (plus précieuses d'abord)."""
    result = []
    for square in chess.SquareSet(board.occupied_co[color]):
        piece = board.piece_at(square)
        if piece is None or piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(not color, square)
        if not attackers:
            continue
        cheapest = min(VALUES[board.piece_type_at(a)] or 100 for a in attackers)
        defended = bool(board.attackers(color, square))
        if not defended or cheapest < VALUES[piece.piece_type]:
            result.append(square)
    return sorted(result, key=lambda s: -VALUES[board.piece_type_at(s)])


def material(board: chess.Board, color: chess.Color) -> int:
    """Différence de matériel (en pions) du point de vue de ``color``."""
    total = 0
    for piece_type, value in VALUES.items():
        total += value * (
            len(board.pieces(piece_type, color)) - len(board.pieces(piece_type, not color))
        )
    return total


def _wins_material(board: chess.Board, move: chess.Move) -> chess.Square | None:
    """Case de la pièce gagnée si ``move`` prend une pièce mal protégée."""
    if not board.is_capture(move) or board.is_en_passant(move):
        return None
    victim = board.piece_type_at(move.to_square)
    attacker = board.piece_type_at(move.from_square)
    if victim is None or attacker is None:
        return None
    defended = bool(board.attackers(not board.turn, move.to_square))
    if not defended or VALUES[victim] > VALUES[attacker]:
        return move.to_square
    return None


def describe_move(board: chess.Board, move: chess.Move) -> str:
    """Ce que fait un coup, en clair (``board`` : position avant le coup)."""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return board.san(move)
    parts = []
    if board.is_castling(move):
        parts.append("met le roi à l'abri en roquant")
    elif board.is_capture(move):
        victim = board.piece_at(move.to_square) or chess.Piece(chess.PAWN, not piece.color)
        parts.append(f"prend {piece_name(victim)} en {chess.square_name(move.to_square)}")
    elif piece.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(
        move.from_square
    ) == (0 if piece.color else 7):
        parts.append("développe une pièce")
    elif piece.piece_type == chess.PAWN and move.to_square in CENTER:
        parts.append("occupe le centre")
    if move.promotion:
        parts.append(f"se promeut en {NAMES[move.promotion][0]}")
    after = board.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        parts.append("et fait échec et mat")
    elif after.is_check():
        parts.append("donne échec")
    targets = [
        s
        for s in after.attacks(move.to_square) & after.occupied_co[not piece.color]
        if after.piece_type_at(s) != chess.KING
        and (
            VALUES[after.piece_type_at(s)] > VALUES[piece.piece_type]
            or not after.attackers(not piece.color, s)
        )
    ]
    if len(targets) >= 2:
        parts.append(
            "attaque à la fois " + " et ".join(_at(after, s) for s in targets[:2]) + " (fourchette)"
        )
    elif targets:
        parts.append(f"attaque {_at(after, targets[0])}")
    saved = [
        s
        for s in hanging(board, piece.color)
        if s != move.from_square and s not in hanging(after, piece.color)
    ]
    if saved and not board.is_capture(move):
        parts.append(f"protège {_at(after, saved[0])}")
    if parts:
        return ", ".join(parts)
    name = NAMES[piece.piece_type][0]
    if piece.piece_type == chess.PAWN:
        return "avance un pion pour gagner de l'espace"
    if _centrality(move.to_square) > _centrality(move.from_square):
        return f"rapproche {'la' if NAMES[piece.piece_type][1] == 'f' else 'le'} {name} du centre"
    return f"replace {'la' if NAMES[piece.piece_type][1] == 'f' else 'le'} {name} en " + (
        chess.square_name(move.to_square)
    )


def _centrality(square: chess.Square) -> float:
    file, rank = chess.square_file(square), chess.square_rank(square)
    return -(abs(file - 3.5) + abs(rank - 3.5))


def idea(board: chess.Board, move: chess.Move) -> str:
    """L'idée d'un coup du joueur, sans en donner la case exacte quand c'est possible."""
    if board.is_castling(move):
        return "roquer pour mettre votre roi à l'abri"
    piece = board.piece_type_at(move.from_square)
    what = describe_move(board, move)
    if piece is None:
        return what
    return f"un coup de {_your(piece)}, qui {what}"


# --- jugement d'un coup ------------------------------------------------------------------


def move_quality(
    loss: int,
    best: Line | None,
    board_before: chess.Board,
    played: chess.Move,
    reply: Line | None = None,
) -> Advice:
    """Jugement du coup joué d'après la perte d'évaluation (centipions).

    ``reply`` : meilleure réponse adverse après le coup joué, pour expliquer une erreur.
    """
    me = board_before.turn
    san = _san(board_before, played)
    if best is None or best.move == played or loss <= 20:
        return Advice("coup", f"{san} : excellent, c'est le meilleur coup.", "bien")
    best_san = _san(board_before, best.move)
    better = f"{best_san}, qui {describe_move(board_before, best.move)}"
    missed_mate = _mate(best, me)
    if missed_mate is not None and missed_mate > 0 and _mate(reply, me) is None:
        return Advice(
            "coup", f"{san} : vous aviez un mat en {missed_mate} avec {best_san} !", "attention"
        )
    why = ""
    if reply is not None:
        after = board_before.copy(stack=False)
        after.push(played)
        if reply.move in after.legal_moves:
            mated = _mate(reply, me)
            if mated is not None and mated < 0:
                why = (
                    f" L'IA peut maintenant mater en {-mated} "
                    f"en commençant par {_san(after, reply.move)}."
                )
            else:
                why = (
                    f" Le problème : l'IA peut répondre {_san(after, reply.move)}, "
                    f"qui {describe_move(after, reply.move)}."
                )
    if loss <= 60:
        return Advice("coup", f"{san} : bon coup. {best_san} était un peu plus précis.", "bien")
    if loss <= 120:
        return Advice(
            "coup",
            f"{san} : imprécision, vous cédez un peu d'avantage. Mieux : {better}.",
            "attention",
        )
    cost = "" if loss >= MATE // 2 else f", vous perdez environ {_pawns(loss / 100)}"
    if loss <= 300:
        return Advice("coup", f"{san} : erreur{cost}.{why} Il fallait {better}.", "danger")
    return Advice("coup", f"{san} est une gaffe{cost} !{why} Il fallait {better}.", "danger")


# --- l'adversaire ----------------------------------------------------------------------


def threat(board: chess.Board, threat_line: Line | None, current: Line | None) -> Advice | None:
    """Menace adverse : son meilleur coup s'il pouvait rejouer, si elle coûte cher."""
    if threat_line is None or current is None:
        return None
    opponent = not board.turn
    null = _null(board)
    if threat_line.move not in null.legal_moves:
        return None
    mate = _mate(threat_line, opponent)
    if mate is not None and mate > 0:
        return Advice(
            "menace",
            f"Elle menace de vous mater en {mate} (en commençant par "
            f"{_san(null, threat_line.move)}) : parez cela d'abord !",
            "danger",
        )
    gain = _cp(threat_line, opponent) - _cp(current, opponent)
    if gain < 80:
        return None
    return Advice(
        "menace",
        f"Elle menace {_san(null, threat_line.move)}, qui {describe_move(null, threat_line.move)}.",
        "attention",
    )


def opponent_intent(board_before: chess.Board, move: chess.Move) -> Advice:
    san = _san(board_before, move)
    return Advice("adversaire", f"L'IA a joué {san} : elle {describe_move(board_before, move)}.")


# --- conseils pour le joueur au trait ------------------------------------------------------


def action(board: chess.Board, lines: tuple[Line, ...], threat_line: Line | None) -> Advice:
    """La priorité du moment, une seule, pour le joueur au trait."""
    me = board.turn
    best = lines[0] if lines else None
    if best is None:
        return Advice("action", "Prenez votre temps : cherchez un coup qui améliore une pièce.")
    hint = " Appuyez sur H pour voir le coup."
    mine = _mate(best, me)
    if mine is not None and mine > 0:
        return Advice(
            "action",
            f"Vous pouvez mater en {mine} ! Cherchez des échecs qui ne laissent "
            f"aucune issue au roi adverse.{hint}",
            "bien",
        )
    if board.is_check():
        return Advice(
            "action",
            f"Vous êtes en échec. Meilleure parade : {idea(board, best.move)}.{hint}",
            "danger",
        )
    threatened = _mate(threat_line, not me)
    if threatened is not None and threatened > 0:
        return Advice(
            "action",
            f"Empêchez le mat : {idea(board, best.move)}.{hint}",
            "danger",
        )
    won = _wins_material(board, best.move)
    if won is not None:
        return Advice(
            "action",
            f"Occasion : {_at(board, won)} est mal "
            f"{_agree(board, won, 'protégé', 'protégée')}, vous pouvez "
            f"{_agree(board, won, 'le', 'la')} gagner.{hint}",
            "bien",
        )
    attacked = hanging(board, me)
    if attacked and best.move.from_square in attacked:
        square = best.move.from_square
        return Advice(
            "action",
            f"Mettez {_mine(board, square)} à l'abri : "
            f"{_agree(board, square, 'il', 'elle')} est attaqué{_agree(board, square, '', 'e')} "
            f"et mal défendu{_agree(board, square, '', 'e')}.{hint}",
            "danger",
        )
    if attacked:
        square = attacked[0]
        return Advice(
            "action",
            f"{_mine(board, square).capitalize()} est en prise, mais le plus fort est "
            f"{idea(board, best.move)}.{hint}",
            "attention",
        )
    if len(lines) >= 2 and _cp(lines[0], me) - _cp(lines[1], me) >= 150 and _cp(lines[0], me) > 0:
        return Advice(
            "action",
            "Un coup est nettement plus fort que les autres : cherchez une tactique "
            f"(échec, prise, attaque double).{hint}",
            "bien",
        )
    found = threat(board, threat_line, best)
    if found is not None:
        null = _null(board)
        return Advice(
            "action",
            f"Neutralisez la menace {_san(null, threat_line.move)}. "
            f"Le plus solide : {idea(board, best.move)}.{hint}",
            "attention",
        )
    return Advice("action", f"Le meilleur plan : {idea(board, best.move)}.{hint}")


def poisoned(board: chess.Board, lines: tuple[Line, ...]) -> Advice | None:
    """Prise tentante mais perdante pour le joueur au trait."""
    me = board.turn
    for move in board.legal_moves:
        if not board.is_capture(move) or any(line.move == move for line in lines):
            continue
        victim = board.piece_type_at(move.to_square) or chess.PAWN
        attacker = board.piece_type_at(move.from_square)
        if (
            attacker
            and VALUES[attacker] > VALUES[victim]
            and board.attackers(not me, move.to_square)
        ):
            return Advice(
                "éviter",
                f"Ne prenez pas {_at(board, move.to_square)} avec {_your(attacker)} "
                f"({_san(board, move)}) : {_agree(board, move.to_square, 'il', 'elle')} est "
                f"{_agree(board, move.to_square, 'défendu', 'défendue')}, vous perdriez "
                f"{_your(attacker)} pour un simple {NAMES[victim][0]}.",
                "attention",
            )
    return None


def strategy(board: chess.Board, me: chess.Color) -> list[Advice]:
    """Conseils de fond selon la phase de jeu et la structure (les plus utiles d'abord)."""
    from chessio.moteur.evaluation import ADJACENT_FILES, FILE_MASKS, PASSED_MASKS, phase

    tips: list[str] = []
    game_phase = phase(board)
    own = board.occupied_co[me]
    king = board.king(me)
    back_rank = chess.BB_RANK_1 if me == chess.WHITE else chess.BB_RANK_8
    undeveloped = chess.popcount((board.knights | board.bishops) & own & back_rank)
    castled = (
        king is not None
        and chess.square_file(king) in (1, 2, 6)
        and chess.square_rank(king) in (0, 7)
    )
    if game_phase >= 18:  # ouverture
        if not chess.SquareSet(board.pawns & own) & CENTER:
            squares = "e4 ou d4" if me == chess.WHITE else "e5 ou d5"
            tips.append(f"Prenez le centre avec un pion ({squares}) : vos pièces sortiront mieux.")
        if undeveloped >= 2:
            tips.append(
                f"Sortez vos cavaliers et vos fous : {undeveloped} sont encore sur leur case "
                "de départ. Évitez de jouer deux fois la même pièce."
            )
        if not castled and board.has_castling_rights(me):
            tips.append("Roquez bientôt : un roi au centre est une cible.")
    elif game_phase >= 6:  # milieu de partie
        if not castled and king is not None and board.queens & board.occupied_co[not me]:
            tips.append("Votre roi est encore au centre : mettez-le à l'abri avant d'attaquer.")
        open_files = [f for f in range(8) if not board.pawns & FILE_MASKS[f]]
        rooks_on_open = any(
            chess.square_file(s) in open_files for s in chess.SquareSet(board.rooks & own)
        )
        if open_files and board.rooks & own and not rooks_on_open:
            tips.append(
                f"Placez une tour sur la colonne ouverte {chess.FILE_NAMES[open_files[0]]} : "
                "elle y sera très active."
            )
        if chess.popcount(board.bishops & own) >= 2:
            tips.append("Vous avez la paire de fous : ouvrez le jeu pour les faire valoir.")
    else:  # finale
        tips.append("Finale : avancez votre roi vers le centre, c'est une pièce forte maintenant.")
    for square in chess.SquareSet(board.pawns & own):
        if not board.pawns & board.occupied_co[not me] & PASSED_MASKS[me][square]:
            tips.append(
                f"Votre pion {chess.square_name(square)} est passé (aucun pion ne peut "
                "l'arrêter) : poussez-le, soutenu par vos pièces."
            )
            break
    for square in chess.SquareSet(board.pawns & board.occupied_co[not me]):
        if not board.pawns & board.occupied_co[not me] & ADJACENT_FILES[chess.square_file(square)]:
            tips.append(
                f"Le pion adverse {chess.square_name(square)} est isolé (aucun pion voisin "
                "pour le défendre) : attaquez-le."
            )
            break
    return [Advice("stratégie", tip) for tip in tips[:3]]


def summary(board: chess.Board, best: Line | None, me: chess.Color) -> tuple[str, str]:
    """Bilan de la position en mots, du point de vue du joueur : (texte, niveau)."""
    if best is None:
        return "", "info"
    mate = _mate(best, me)
    if mate is not None:
        if mate > 0:
            return f"Vous avez un mat en {mate} !", "bien"
        return f"L'IA a un mat en {-mate} contre vous.", "danger"
    cp = _cp(best, me)
    if cp > 400:
        text, level = "Vous êtes gagnant", "bien"
    elif cp > 150:
        text, level = "Vous avez l'avantage", "bien"
    elif cp > 50:
        text, level = "Vous êtes un peu mieux", "bien"
    elif cp >= -50:
        text, level = "Position équilibrée", "info"
    elif cp >= -150:
        text, level = "L'IA est un peu mieux", "attention"
    elif cp >= -400:
        text, level = "L'IA a l'avantage", "danger"
    else:
        text, level = "Vous êtes en grande difficulté", "danger"
    balance = material(board, me)
    if balance == 0:
        stuff = "matériel égal"
    elif balance > 0:
        stuff = f"vous avez {balance} point{'s' if balance > 1 else ''} de matériel en plus"
    else:
        stuff = f"il vous manque {-balance} point{'s' if balance < -1 else ''} de matériel"
    return f"{text} · {stuff}", level


def advise(board: chess.Board, lines: tuple[Line, ...], threat_line: Line | None) -> list[Advice]:
    """Conseils hiérarchisés pour le joueur au trait, après le dernier coup de l'IA."""
    me = board.turn
    advices: list[Advice] = []
    best = lines[0] if lines else None
    found = threat(board, threat_line, best)
    if board.move_stack:
        before = board.copy()
        last = before.pop()
        intent = opponent_intent(before, last)
        text, level = intent.text, intent.level
        if found is not None:
            text, level = f"{text} {found.text}", found.level
        advices.append(Advice("adversaire", text, level))
    elif found is not None:
        advices.append(found)
    now = action(board, lines, threat_line)
    advices.append(now)
    avoid = poisoned(board, lines)
    if avoid is not None:
        advices.append(avoid)
    if now.level in ("info", "bien") and avoid is None:
        tips = strategy(board, me)
        if tips:
            advices.append(Advice("idée", tips[0].text))
    return advices

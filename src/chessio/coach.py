"""Assistant d'entraînement : explications en français pendant une partie contre l'IA.

Les textes sont produits par des règles d'échecs appuyées sur les calculs du
moteur (aucun service extérieur) :

* qualité du dernier coup du joueur (perte d'évaluation) et meilleure alternative ;
* intention du dernier coup adverse et **menace** (ce que l'adversaire jouerait
  s'il rejouait : analyse après un « coup nul ») ;
* **pièges** : pièces en prise, prises empoisonnées, mat en vue ;
* **stratégie** selon la phase de jeu et la structure, et plan du moteur.
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


@dataclass(frozen=True)
class Advice:
    kind: str  # « coup », « adversaire », « menace », « piège », « stratégie », « plan »
    text: str
    level: str = "info"  # « bien », « info », « attention », « danger »


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


def _cp(line: Line, color: chess.Color) -> int:
    """Évaluation (centipions) du point de vue de ``color`` ; mat ramené à ±10 000."""
    score = line.score.pov(color)
    return score.score(mate_score=10_000)


def hanging(board: chess.Board, color: chess.Color) -> list[chess.Square]:
    """Pièces de ``color`` attaquées et insuffisamment défendues."""
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
    return result


def describe_move(board: chess.Board, move: chess.Move) -> str:
    """Ce que fait un coup, en clair (``board`` : position avant le coup)."""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return board.san(move)
    parts = []
    if board.is_castling(move):
        parts.append("met son roi à l'abri en roquant")
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
    return ", ".join(parts) if parts else "améliore la position d'une pièce"


def move_quality(
    loss: int, best: Line | None, board_before: chess.Board, played: chess.Move
) -> Advice:
    """Jugement du coup joué d'après la perte d'évaluation (centipions)."""
    san = _san(board_before, played)
    if best is None or best.move == played or loss <= 20:
        return Advice("coup", f"{san} : excellent coup, c'est le choix du moteur.", "bien")
    alt = f"Mieux : {_san(board_before, best.move)}, qui {describe_move(board_before, best.move)}."
    if loss <= 60:
        return Advice("coup", f"{san} : bon coup. {alt}", "bien")
    if loss <= 120:
        return Advice("coup", f"{san} : imprécision (−{loss / 100:.1f}). {alt}", "attention")
    if loss <= 300:
        return Advice("coup", f"{san} : erreur (−{loss / 100:.1f}). {alt}", "danger")
    return Advice("coup", f"{san} : gaffe (−{loss / 100:.1f}) ! {alt}", "danger")


def threat(board: chess.Board, threat_line: Line | None, current: Line | None) -> Advice | None:
    """Menace adverse : son meilleur coup s'il pouvait rejouer, si elle coûte cher."""
    if threat_line is None or current is None:
        return None
    opponent = not board.turn
    null = board.copy(stack=False)
    null.turn = not null.turn  # « coup nul » : l'adversaire rejoue
    null.ep_square = None
    gain = _cp(threat_line, opponent) - _cp(current, opponent)
    mate = threat_line.score.pov(opponent).mate()
    if mate is not None and mate > 0:
        return Advice(
            "menace",
            f"Danger : l'adversaire menace mat en {mate} ({_san(null, threat_line.move)}…). "
            "Parez-le d'abord !",
            "danger",
        )
    if gain < 80:
        return None
    return Advice(
        "menace",
        f"Menace : l'adversaire veut jouer {_san(null, threat_line.move)}, "
        f"qui {describe_move(null, threat_line.move)}.",
        "attention",
    )


def traps(board: chess.Board, me: chess.Color, lines: tuple[Line, ...]) -> list[Advice]:
    """Pièces en prise et prises empoisonnées pour le joueur ``me`` (au trait)."""
    advices = []
    for square in hanging(board, me)[:2]:
        advices.append(
            Advice(
                "piège",
                f"{_at(board, square).capitalize()} est en prise : "
                + _agree(board, square, "protégez-le", "protégez-la")
                + " ou "
                + _agree(board, square, "déplacez-le.", "déplacez-la."),
                "danger",
            )
        )
    if lines:
        best = _cp(lines[0], me)
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
                advices.append(
                    Advice(
                        "piège",
                        f"Piège : {_san(board, move)} paraît tentant, mais "
                        f"{_at(board, move.to_square)} est "
                        f"{_agree(board, move.to_square, 'défendu', 'défendue')} — vous perdriez "
                        "du matériel.",
                        "attention",
                    )
                )
                break
        if best > 200:
            advices.append(
                Advice(
                    "plan",
                    f"Vous êtes nettement mieux (+{best / 100:.1f}) : cherchez à simplifier.",
                    "bien",
                )
            )
    return advices


def strategy(board: chess.Board, me: chess.Color) -> list[Advice]:
    """Conseils généraux selon la phase de jeu et la structure."""
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
        if undeveloped >= 2:
            tips.append(
                f"Ouverture : sortez vos pièces mineures ({undeveloped} encore à leur place)."
            )
        if not castled and board.has_castling_rights(me):
            tips.append("Mettez votre roi à l'abri en roquant bientôt.")
        if not chess.SquareSet(board.pawns & own) & CENTER:
            tips.append("Disputez le centre avec un pion (d4/e4 ou d5/e5).")
    elif game_phase >= 6:  # milieu de partie
        for square in chess.SquareSet(board.rooks & own):
            mask = FILE_MASKS[chess.square_file(square)]
            if board.pawns & own & mask:
                open_files = [f for f in range(8) if not board.pawns & FILE_MASKS[f]]
                if open_files:
                    tips.append(
                        f"Placez une tour sur la colonne ouverte {chess.FILE_NAMES[open_files[0]]}."
                    )
                break
        if not castled and king is not None:
            tips.append("Votre roi est encore au centre : sa sécurité passe avant tout.")
        if chess.popcount(board.bishops & own) >= 2:
            tips.append("Vous avez la paire de fous : ouvrez le jeu pour les faire valoir.")
    else:  # finale
        tips.append("Finale : activez votre roi, c'est une pièce forte maintenant.")
    for square in chess.SquareSet(board.pawns & own):
        if not board.pawns & board.occupied_co[not me] & PASSED_MASKS[me][square]:
            tips.append(f"Votre pion {chess.square_name(square)} est passé : soutenez son avance.")
            break
    for square in chess.SquareSet(board.pawns & board.occupied_co[not me]):
        if not board.pawns & board.occupied_co[not me] & ADJACENT_FILES[chess.square_file(square)]:
            tips.append(
                f"Le pion adverse {chess.square_name(square)} est isolé : une cible à attaquer."
            )
            break
    return [Advice("stratégie", tip) for tip in tips[:3]]


def plan(board: chess.Board, line: Line | None) -> Advice | None:
    if line is None or len(line.pv) < 2:
        return None
    from chessio.ui.notation import figurine_line

    return Advice("plan", f"Plan du moteur : {figurine_line(board, line.pv[:6])}")


def opponent_intent(board_before: chess.Board, move: chess.Move) -> Advice:
    san = _san(board_before, move)
    return Advice("adversaire", f"L'IA a joué {san} : elle {describe_move(board_before, move)}.")

"""Recherche du meilleur coup : négamax alpha-bêta (lot L3).

* approfondissement itératif, limité par le temps, la profondeur ou les nœuds ;
* table de transposition (scores de mat ramenés au nœud stocké) ;
* recherche de quiescence sur les prises et promotions, pour ne pas évaluer une
  position au milieu d'un échange ;
* ordre des coups : coup de la table, prises MVV-LVA, promotions, coups
  « tueurs », puis historique ;
* MultiPV : les ``k`` meilleurs coups, en excluant à la racine ceux déjà trouvés ;
* v0.2 : recherche à fenêtre nulle (PVS), fenêtres d'aspiration, coup nul,
  réductions des coups tardifs (LMR), élagages statique et de frontière.

Le générateur de coups est celui de python-chess ; la v2 en Rust (lot L4)
reprendra cette logique avec ses propres bitboards.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import chess

from chessio.moteur.evaluation import evaluate

INF = 1_000_000
MATE = 100_000
MATE_BOUND = MATE - 1_000  # au-delà : score de mat
MAX_PLY = 100
EXACT, LOWER, UPPER = 0, 1, 2
DEFAULT_TT_ENTRIES = 400_000  # environ 150 Mo en Python
CHECK_EVERY = 1023  # masque : contrôle du temps tous les 1 024 nœuds
DELTA_MARGIN = 200  # quiescence : prise jugée inutile si même gagnée elle ne suffit pas
ASPIRATION = 40  # demi-largeur de la fenêtre d'aspiration à la racine (centipions)
REVERSE_FUTILITY = 120  # élagage statique : marge par demi-coup restant
FUTILITY_MARGIN = 150  # élagage de frontière : marge par demi-coup restant

# Valeurs pour l'ordre des coups (le roi « attaquant » passe en dernier).
ORDER_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 2000,
}


class SearchAborted(Exception):
    """Temps écoulé, limite de nœuds atteinte ou arrêt demandé."""


@dataclass
class Limits:
    """Limites d'une recherche (temps en secondes), comme la commande UCI « go »."""

    depth: int | None = None
    movetime: float | None = None
    nodes: int | None = None
    wtime: float | None = None
    btime: float | None = None
    winc: float = 0.0
    binc: float = 0.0
    movestogo: int | None = None
    infinite: bool = False

    def budget(self, turn: chess.Color) -> tuple[float | None, float | None]:
        """(temps souple, temps dur) : pas de nouvelle itération après le premier,
        interruption au second. ``(None, None)`` : pas de limite de temps."""
        if self.infinite:
            return None, None
        if self.movetime is not None:
            return 0.6 * self.movetime, self.movetime
        remaining = self.wtime if turn == chess.WHITE else self.btime
        if remaining is None:
            return None, None
        increment = self.winc if turn == chess.WHITE else self.binc
        soft = remaining / (self.movestogo or 30) + 0.75 * increment
        hard = max(0.01, min(0.4 * remaining, 3 * soft) - 0.05)
        return min(soft, hard), hard


@dataclass(frozen=True)
class SearchLine:
    score: int  # centipions du point de vue du camp au trait, ou score de mat
    moves: tuple[chess.Move, ...]


@dataclass(frozen=True)
class SearchInfo:
    depth: int
    lines: tuple[SearchLine, ...]
    nodes: int
    elapsed: float

    @property
    def best(self) -> chess.Move | None:
        return self.lines[0].moves[0] if self.lines and self.lines[0].moves else None


def mate_in(score: int) -> int | None:
    """Coups (et non demi-coups) avant le mat : positif si l'on mate, négatif si l'on est maté."""
    if abs(score) < MATE_BOUND:
        return None
    plies = MATE - abs(score)
    moves = (plies + 1) // 2
    return moves if score > 0 else -moves


class Searcher:
    def __init__(self, tt_entries: int = DEFAULT_TT_ENTRIES) -> None:
        self.tt: dict[object, tuple[int, int, int, chess.Move | None]] = {}
        self.tt_entries = tt_entries
        self.history: dict[tuple[bool, int, int], int] = {}
        self.stop_requested = False  # peut être levé depuis un autre fil
        self.nodes = 0
        self._deadline: float | None = None
        self._node_limit: int | None = None
        self._can_abort = False
        self._killers: list[list[chess.Move | None]] = []
        self._pv: list[list[chess.Move]] = []

    def clear(self) -> None:
        """Oublie tout (nouvelle partie)."""
        self.tt.clear()
        self.history.clear()

    def stop(self) -> None:
        self.stop_requested = True

    # --- approfondissement itératif ------------------------------------------

    def search(
        self,
        board: chess.Board,
        limits: Limits,
        *,
        multipv: int = 1,
        on_iteration: Callable[[SearchInfo], None] | None = None,
    ) -> SearchInfo:
        """Cherche les ``multipv`` meilleurs coups dans les limites données."""
        board = board.copy()  # avec l'historique : utile à la détection des répétitions
        start = time.perf_counter()
        self.stop_requested = False
        self.nodes = 0
        soft, hard = limits.budget(board.turn)
        self._deadline = None if hard is None else start + hard
        self._node_limit = limits.nodes
        self._killers = [[None, None] for _ in range(MAX_PLY + 2)]
        self._pv = [[] for _ in range(MAX_PLY + 2)]

        root_moves = list(board.legal_moves)
        if not root_moves:
            return SearchInfo(0, (), 0, 0.0)
        multipv = max(1, min(multipv, len(root_moves)))
        result: SearchInfo | None = None
        previous: list[int] = []  # scores de l'itération précédente, ligne par ligne
        for depth in range(1, (limits.depth or MAX_PLY) + 1):
            self._can_abort = depth > 1  # la profondeur 1 va toujours à son terme
            try:
                lines: list[SearchLine] = []
                excluded: list[chess.Move] = []
                for index in range(multipv):
                    guess = previous[index] if index < len(previous) else None
                    score, pv = self._root(board, depth, excluded, guess)
                    lines.append(SearchLine(score, tuple(pv)))
                    excluded.append(pv[0])
            except SearchAborted:
                break
            previous = [line.score for line in lines]
            lines.sort(key=lambda line: line.score, reverse=True)
            result = SearchInfo(depth, tuple(lines), self.nodes, time.perf_counter() - start)
            if on_iteration is not None:
                on_iteration(result)
            if self.stop_requested:
                break
            if soft is not None and result.elapsed >= soft:
                break
            if self._node_limit is not None and self.nodes >= self._node_limit:
                break
            mate = mate_in(lines[0].score)
            if not limits.infinite and mate is not None and depth >= 2 * abs(mate) + 1:
                break  # mat trouvé et confirmé : inutile de chercher plus loin
        assert result is not None  # la profondeur 1 ne peut pas être interrompue
        return result

    def _tick(self) -> None:
        if not self._can_abort:
            return
        if self.stop_requested:
            raise SearchAborted
        if self._deadline is not None and time.perf_counter() >= self._deadline:
            raise SearchAborted
        if self._node_limit is not None and self.nodes >= self._node_limit:
            raise SearchAborted

    # --- racine -------------------------------------------------------------

    def _root(
        self, board: chess.Board, depth: int, excluded: list[chess.Move], guess: int | None
    ) -> tuple[int, list[chess.Move]]:
        """Fenêtre d'aspiration autour du score précédent, élargie en cas d'échec."""
        if guess is None or depth < 4 or abs(guess) >= MATE_BOUND:
            return self._root_window(board, depth, excluded, -INF, INF)
        window = ASPIRATION
        alpha, beta = guess - window, guess + window
        while True:
            score, pv = self._root_window(board, depth, excluded, alpha, beta)
            if score <= alpha and alpha > -INF:
                window *= 4
                alpha = max(-INF, score - window)
            elif score >= beta and beta < INF:
                window *= 4
                beta = min(INF, score + window)
            else:
                return score, pv

    def _root_window(
        self, board: chess.Board, depth: int, excluded: list[chess.Move], alpha: int, beta: int
    ) -> tuple[int, list[chess.Move]]:
        original_alpha = alpha
        best_score, best_move = -INF, None
        key = board._transposition_key()
        entry = self.tt.get(key)
        tt_move = entry[3] if entry is not None else None
        best_pv: list[chess.Move] = []
        first = True
        for move in self._ordered(board, board.legal_moves, tt_move, 0):
            if move in excluded:
                continue
            board.push(move)
            try:
                if first:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
                else:
                    # Fenêtre nulle : ce coup fait-il mieux que le meilleur actuel ?
                    score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                    if alpha < score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            finally:
                board.pop()
            first = False
            if score > best_score:
                best_score, best_move = score, move
                best_pv = [move, *self._pv[1]]
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        if not excluded and original_alpha < best_score < beta:
            self._store(key, depth, best_score, EXACT, best_move, 0)
        return best_score, self._complete_pv(board, best_pv, depth)

    def _complete_pv(
        self, board: chess.Board, pv: list[chess.Move], depth: int
    ) -> list[chess.Move]:
        """Prolonge la variante principale avec la table de transposition."""
        pv = list(pv)
        pushed = 0
        try:
            for move in pv:
                board.push(move)
                pushed += 1
            while len(pv) < depth and not board.is_game_over(claim_draw=False):
                entry = self.tt.get(board._transposition_key())
                if entry is None or entry[3] is None or not board.is_legal(entry[3]):
                    break
                pv.append(entry[3])
                board.push(entry[3])
                pushed += 1
        finally:
            for _ in range(pushed):
                board.pop()
        return pv

    # --- négamax --------------------------------------------------------------

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null: bool = True,
    ) -> int:
        self.nodes += 1
        if not self.nodes & CHECK_EVERY:
            self._tick()
        self._pv[ply] = []

        # Nulles : règle des 50 coups, répétition (une seule suffit dans la recherche).
        if board.halfmove_clock >= 100 or board.is_repetition(2):
            return 0
        # Élagage par distance au mat : inutile de chercher un mat plus lent qu'un mat déjà trouvé.
        alpha = max(alpha, -MATE + ply)
        beta = min(beta, MATE - ply - 1)
        if alpha >= beta:
            return alpha

        in_check = board.is_check()
        if in_check:
            depth += 1  # extension d'échec : ne pas s'arrêter au milieu d'une attaque
        if depth <= 0 or ply >= MAX_PLY:
            return self._quiescence(board, alpha, beta, ply)

        pv_node = beta - alpha > 1
        key = board._transposition_key()
        entry = self.tt.get(key)
        tt_move = None
        if entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = entry
            if tt_depth >= depth and not pv_node:
                score = self._from_tt(tt_score, ply)
                if (
                    tt_flag == EXACT
                    or (tt_flag == LOWER and score >= beta)
                    or (tt_flag == UPPER and score <= alpha)
                ):
                    return score

        static_eval = None
        if not in_check and not pv_node:
            static_eval = evaluate(board)
            # Élagage statique : si loin au-dessus de bêta, l'adversaire évitera cette position.
            if (
                depth <= 3
                and abs(beta) < MATE_BOUND
                and static_eval - REVERSE_FUTILITY * depth >= beta
            ):
                return static_eval
            # Coup nul : si passer son tour suffit encore à dépasser bêta, inutile de chercher.
            if allow_null and depth >= 3 and static_eval >= beta and self._has_pieces(board):
                reduction = 3 if depth >= 6 else 2
                board.push(chess.Move.null())
                try:
                    score = -self._negamax(
                        board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, False
                    )
                finally:
                    board.pop()
                if score >= beta:
                    return beta if score >= MATE_BOUND else score
        # Élagage de frontière : près des feuilles, un coup tranquille ne rattrapera pas alpha.
        futile = (
            static_eval is not None
            and depth <= 2
            and static_eval + FUTILITY_MARGIN * depth <= alpha
        )

        original_alpha = alpha
        best_score, best_move = -INF, None
        searched = 0
        killers = self._killers[ply]
        for move in self._ordered(board, board.legal_moves, tt_move, ply):
            quiet = move.promotion is None and not board.is_capture(move)
            late = searched >= 3 and depth >= 3 and not in_check
            gives_check = (
                board.gives_check(move) if quiet and searched and (futile or late) else False
            )
            if futile and quiet and searched and not gives_check:
                continue
            # Réduction des coups tardifs : bien ordonnés, les derniers coups tranquilles
            # sont rarement les meilleurs ; on les cherche moins loin, quitte à recommencer.
            reduction = 0
            if late and quiet and not gives_check and move not in killers:
                reduction = 1 if searched < 6 or pv_node else 2
            board.push(move)
            try:
                if not searched:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self._negamax(
                        board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1
                    )
                    if reduction and score > alpha:
                        score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                    if pv_node and alpha < score < beta:
                        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            searched += 1
            if score > best_score:
                best_score, best_move = score, move
                if score > alpha:
                    alpha = score
                    self._pv[ply] = [move, *self._pv[ply + 1]]
                    if alpha >= beta:
                        if not board.is_capture(move) and move.promotion is None:
                            self._remember_quiet(board, move, depth, ply)
                        break
        if searched == 0:
            return -MATE + ply if in_check else 0  # mat ou pat

        if best_score <= original_alpha:
            flag = UPPER
        elif best_score >= beta:
            flag = LOWER
        else:
            flag = EXACT
        self._store(key, depth, best_score, flag, best_move, ply)
        return best_score

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if not self.nodes & CHECK_EVERY:
            self._tick()
        self._pv[ply] = []
        in_check = board.is_check()
        if ply >= MAX_PLY:
            return evaluate(board)

        if in_check:
            # En échec, pas de « statu quo » possible : toutes les parades.
            moves = self._ordered(board, board.legal_moves, None, ply)
            best = -MATE + ply
        else:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            best = stand_pat
            moves = self._ordered(board, self._noisy_moves(board), None, ply)

        for move in moves:
            if not in_check and move.promotion is None:
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                if best + ORDER_VALUES[victim] + DELTA_MARGIN < alpha:
                    continue  # même gagnée, cette prise ne rattraperait pas alpha
            board.push(move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    @staticmethod
    def _has_pieces(board: chess.Board) -> bool:
        """Le camp au trait a-t-il des pièces ? (sinon zugzwang probable : pas de coup nul)"""
        pieces = board.knights | board.bishops | board.rooks | board.queens
        return bool(pieces & board.occupied_co[board.turn])

    @staticmethod
    def _noisy_moves(board: chess.Board) -> Iterable[chess.Move]:
        """Prises et promotions en dame : les coups qui changent le matériel."""
        yield from board.generate_legal_captures()
        own_pawns = board.pawns & board.occupied_co[board.turn]
        for move in board.generate_legal_moves(own_pawns, chess.BB_BACKRANKS & ~board.occupied):
            if move.promotion == chess.QUEEN:
                yield move

    # --- ordre des coups ----------------------------------------------------------

    def _ordered(
        self,
        board: chess.Board,
        moves: Iterable[chess.Move],
        tt_move: chess.Move | None,
        ply: int,
    ) -> list[chess.Move]:
        killers = self._killers[ply] if ply < len(self._killers) else (None, None)
        turn = board.turn
        history = self.history
        scored = []
        for move in moves:
            if move == tt_move:
                score = 30_000_000
            elif board.is_capture(move):
                victim = board.piece_type_at(move.to_square) or chess.PAWN  # en passant
                attacker = board.piece_type_at(move.from_square)
                score = 20_000_000 + 10 * ORDER_VALUES[victim] - ORDER_VALUES[attacker]
            elif move.promotion is not None:
                score = 19_000_000 + ORDER_VALUES[move.promotion]
            elif move == killers[0]:
                score = 18_000_000
            elif move == killers[1]:
                score = 17_000_000
            else:
                score = history.get((turn, move.from_square, move.to_square), 0)
            scored.append((score, move))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored]

    def _remember_quiet(self, board: chess.Board, move: chess.Move, depth: int, ply: int) -> None:
        """Coup tranquille ayant provoqué une coupure : tueur à ce niveau, historique ailleurs."""
        killers = self._killers[ply]
        if move != killers[0]:
            killers[1], killers[0] = killers[0], move
        key = (board.turn, move.from_square, move.to_square)
        self.history[key] = min(self.history.get(key, 0) + depth * depth, 1_000_000)

    # --- table de transposition ---------------------------------------------

    def _store(
        self, key: object, depth: int, score: int, flag: int, move: chess.Move | None, ply: int
    ) -> None:
        if len(self.tt) >= self.tt_entries:
            self.tt.clear()  # simple et suffisant à la vitesse d'un moteur Python
        self.tt[key] = (depth, self._to_tt(score, ply), flag, move)

    @staticmethod
    def _to_tt(score: int, ply: int) -> int:
        """Score de mat compté depuis le nœud stocké, et non depuis la racine."""
        if score >= MATE_BOUND:
            return score + ply
        if score <= -MATE_BOUND:
            return score - ply
        return score

    @staticmethod
    def _from_tt(score: int, ply: int) -> int:
        if score >= MATE_BOUND:
            return score - ply
        if score <= -MATE_BOUND:
            return score + ply
        return score

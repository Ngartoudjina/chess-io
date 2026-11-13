"""Moteur maison : v1 en Python (règles, évaluation, recherche, UCI) et v2 en Rust."""

import subprocess
import sys

import chess
import chess.engine
import numpy as np
import pytest

from chessio.engine import find_rust_engine
from chessio.moteur.evaluation import evaluate
from chessio.moteur.perft import divide, perft
from chessio.moteur.search import MATE_BOUND, Limits, Searcher, mate_in
from chessio.moteur.uci import UciEngine, parse_limits, uci_score
from chessio.vision.synthetic import random_position

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
HOMEMADE = [sys.executable, "-m", "chessio.moteur"]


class TestPerft:
    """Valeurs de référence (Chessprogramming Wiki, « Perft Results »)."""

    @pytest.mark.parametrize(("depth", "nodes"), [(1, 20), (2, 400), (3, 8_902), (4, 197_281)])
    def test_start_position(self, depth, nodes):
        assert perft(chess.Board(), depth) == nodes

    @pytest.mark.parametrize(("depth", "nodes"), [(1, 48), (2, 2_039), (3, 97_862)])
    def test_kiwipete(self, depth, nodes):
        # Roques, prises en passant, promotions, clouages : la position piège classique.
        assert perft(chess.Board(KIWIPETE), depth) == nodes

    def test_divide_sums_to_perft(self):
        assert sum(divide(chess.Board(), 3).values()) == 8_902


class TestEvaluation:
    def test_start_position_is_balanced(self):
        assert evaluate(chess.Board()) == 0

    def test_material(self):
        assert evaluate(chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")) > 800
        assert evaluate(chess.Board("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")) < -800

    def test_color_symmetry(self):
        # Une position et son miroir coloré ont exactement le même score pour le camp au trait.
        rng = np.random.default_rng(3)
        for _ in range(50):
            board = random_position(rng)
            assert evaluate(board) == evaluate(board.mirror())

    def test_pawn_advance_counts_more_in_endgame(self):
        far = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        near = chess.Board("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1")
        assert evaluate(near) > evaluate(far) + 50


def best_move(fen: str, depth: int) -> tuple[chess.Move, int]:
    info = Searcher().search(chess.Board(fen), Limits(depth=depth))
    return info.best, info.lines[0].score


class TestSearch:
    def test_mate_in_one(self):
        move, score = best_move("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 2)
        assert move == chess.Move.from_uci("a1a8")
        assert mate_in(score) == 1

    def test_mate_in_two(self):
        # Legs de Philidor : 1. Dg8+ Txg8 2. Cf7#, mat à l'étouffée.
        move, score = best_move("5r1k/6pp/7N/3Q4/8/8/8/6K1 w - - 0 1", 4)
        assert move == chess.Move.from_uci("d5g8")
        assert mate_in(score) == 2

    def test_wins_hanging_queen(self):
        move, score = best_move("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", 3)
        assert move == chess.Move.from_uci("d2d5")
        assert score > 300

    def test_does_not_take_a_defended_pawn_with_the_queen(self):
        # Dxd5 perd la dame contre exd5 : la quiescence doit le voir.
        move, _ = best_move("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1", 2)
        assert move != chess.Move.from_uci("d1d5")

    def test_avoids_stalemate_and_mates(self):
        # Df7 pat les Noirs ; Df8 les mate.
        move, score = best_move("7k/8/6K1/8/8/8/8/5Q2 w - - 0 1", 3)
        assert move == chess.Move.from_uci("f1f8")
        assert mate_in(score) == 1

    def test_no_move_when_stalemated(self):
        info = Searcher().search(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), Limits(depth=3))
        assert info.lines == () and info.best is None

    def test_repetition_counts_as_draw(self):
        # Les Blancs ont une dame de plus. Après Cf3 Cf6 Cg1 Cg8, rejouer Cf3 recrée une
        # position déjà vue : la recherche la compte nulle (0), pas +900.
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push_uci(uci)
        info = Searcher().search(board, Limits(depth=1), multipv=40)
        scores = {line.moves[0].uci(): line.score for line in info.lines}
        assert scores["g1f3"] == 0
        assert max(scores.values()) > 500

    def test_multipv_returns_distinct_ordered_moves(self):
        info = Searcher().search(chess.Board(), Limits(depth=3), multipv=3)
        moves = [line.moves[0] for line in info.lines]
        assert len(set(moves)) == 3
        scores = [line.score for line in info.lines]
        assert scores == sorted(scores, reverse=True)

    def test_respects_movetime(self):
        info = Searcher().search(chess.Board(KIWIPETE), Limits(movetime=0.5))
        assert info.elapsed < 1.0
        assert info.best in chess.Board(KIWIPETE).legal_moves

    def test_node_limit(self):
        searcher = Searcher()
        searcher.search(chess.Board(), Limits(nodes=5_000))
        assert searcher.nodes < 15_000

    def test_transposition_table_saves_work(self):
        searcher = Searcher()
        searcher.search(chess.Board(KIWIPETE), Limits(depth=3))
        first = searcher.nodes
        searcher.search(chess.Board(KIWIPETE), Limits(depth=3))
        assert searcher.nodes < first


class TestUci:
    def test_score_format(self):
        assert uci_score(35) == "cp 35"
        assert uci_score(MATE_BOUND + 997) == "mate 2"  # MATE − 3 demi-coups
        assert uci_score(-(MATE_BOUND + 998)) == "mate -1"

    def test_parse_limits(self):
        limits = parse_limits("wtime 60000 btime 50000 winc 1000 movestogo 20".split())
        assert (limits.wtime, limits.btime, limits.winc, limits.movestogo) == (60, 50, 1, 20)
        assert parse_limits(["infinite"]).infinite

    def test_position_and_perft_commands(self):
        lines = []

        class Out:
            def write(self, text):
                lines.append(text)

            def flush(self):
                pass

        engine = UciEngine(out=Out())
        engine.handle("position startpos moves e2e4 e7e5")
        assert engine.board.fen().startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w")
        engine.handle("go perft 2")
        expected = perft(engine.board.copy(), 2)
        assert any(f"Nodes searched: {expected}" in line for line in lines)

    def test_through_python_chess(self):
        """Le moteur se pilote comme n'importe quel moteur UCI (ce que fait l'application)."""
        with chess.engine.SimpleEngine.popen_uci(HOMEMADE) as engine:
            assert engine.id["name"].startswith("Chessio Moteur")
            board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
            result = engine.play(board, chess.engine.Limit(time=0.5))
            assert result.move == chess.Move.from_uci("a1a8")
            infos = engine.analyse(chess.Board(), chess.engine.Limit(time=0.5), multipv=2)
            assert len(infos) == 2
            assert infos[0]["pv"][0] != infos[1]["pv"][0]


rust_engine = pytest.mark.skipif(
    find_rust_engine() is None, reason="moteur Rust non compilé (cargo build --release)"
)


@rust_engine
class TestRustEngine:
    """Moteur maison v2 (lot L4), piloté en UCI comme dans l'application."""

    @pytest.fixture(scope="class")
    def engine(self):
        with chess.engine.SimpleEngine.popen_uci([str(find_rust_engine())]) as engine:
            yield engine

    def test_identity(self, engine):
        assert engine.id["name"].startswith("Chessio Moteur 0.3")

    @pytest.mark.parametrize(
        ("fen", "depth", "nodes"),
        [(chess.STARTING_FEN, 5, 4_865_609), (KIWIPETE, 4, 4_085_603)],
    )
    def test_perft(self, fen, depth, nodes):
        script = f"position fen {fen}\ngo perft {depth}\nquit\n"
        out = subprocess.run(
            [str(find_rust_engine())], input=script, capture_output=True, text=True, timeout=60
        ).stdout
        assert f"Nodes searched: {nodes}" in out

    def test_same_evaluation_as_python(self):
        rng = np.random.default_rng(8)
        fens = [random_position(rng).fen() for _ in range(50)]
        script = "".join(f"position fen {fen}\neval\n" for fen in fens) + "quit\n"
        out = subprocess.run(
            [str(find_rust_engine())], input=script, capture_output=True, text=True, timeout=60
        ).stdout.split()
        rust = [int(value) for key, value in zip(out[::2], out[1::2], strict=True) if key == "eval"]
        assert rust == [evaluate(chess.Board(fen)) for fen in fens]

    def test_mates(self, engine):
        board = chess.Board("5r1k/6pp/7N/3Q4/8/8/8/6K1 w - - 0 1")  # legs de Philidor
        info = engine.analyse(board, chess.engine.Limit(time=0.5))
        assert info["pv"][0] == chess.Move.from_uci("d5g8")
        assert info["score"].white().mate() == 2

    def test_avoids_stalemate(self, engine):
        board = chess.Board("7k/8/6K1/8/8/8/8/5Q2 w - - 0 1")
        assert engine.play(board, chess.engine.Limit(time=0.3)).move == chess.Move.from_uci("f1f8")

    def test_repetition_is_a_draw(self, engine):
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push_uci(uci)
        infos = engine.analyse(board, chess.engine.Limit(depth=3), multipv=30)
        scores = {info["pv"][0]: info["score"].relative.score() for info in infos}
        assert scores[chess.Move.from_uci("g1f3")] == 0

    def test_multipv_and_speed(self, engine):
        infos = engine.analyse(chess.Board(KIWIPETE), chess.engine.Limit(time=1.0), multipv=3)
        assert len({info["pv"][0] for info in infos}) == 3
        assert infos[0]["depth"] >= 6  # le Python v0.2 atteint 3 à 4
        assert infos[0]["nps"] > 200_000

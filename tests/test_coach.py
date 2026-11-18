"""Assistant d'entraînement : pièces en prise, description des coups, menaces, conseils."""

import chess
import pytest

from chessio import coach
from chessio.engine import Engine, EngineNotFoundError, find_stockfish


@pytest.fixture(scope="module")
def engine():
    try:
        find_stockfish()
    except EngineNotFoundError:
        pytest.skip("Stockfish absent")
    with Engine(threads=1, hash_mb=16) as eng:
        yield eng


def test_hanging_piece():
    board = chess.Board("4k3/8/4p3/3Q4/8/8/8/4K3 w - - 0 1")  # dame attaquée par un pion
    assert chess.D5 in coach.hanging(board, chess.WHITE)


def test_defended_piece_attacked_by_equal_is_not_hanging():
    board = chess.Board("4k3/8/2n5/3N4/2P5/8/8/4K3 w - - 0 1")
    assert chess.D5 not in coach.hanging(board, chess.WHITE)


def test_describe_capture_and_check():
    board = chess.Board("4k3/8/8/8/8/8/4q3/R3K3 w - - 0 1")
    text = coach.describe_move(board, chess.Move.from_uci("e1e2"))
    assert "prend la dame noire en e2" in text


def test_describe_fork():
    board = chess.Board("r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1")
    text = coach.describe_move(board, chess.Move.from_uci("b5c7"))
    assert "donne échec" in text and "la tour noire en a8" in text


def test_move_quality_grades(engine):
    board = chess.Board()
    best = engine.analyse(board, time_limit=0.2).best
    assert coach.move_quality(0, best, board, best.move).level == "bien"
    blunder = coach.move_quality(500, best, board, chess.Move.from_uci("f2f3"))
    assert blunder.level == "danger" and "gaffe" in blunder.text and best.san[-2:] in blunder.text


def test_threat_of_back_rank_mate(engine):
    board = chess.Board("r5k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1")
    current = engine.analyse(board, time_limit=0.3).best
    null = board.copy(stack=False)
    null.turn = not null.turn
    threat_line = engine.analyse(null, time_limit=0.3).best
    advice = coach.threat(board, threat_line, current)
    assert advice is not None and advice.level == "danger" and "mat" in advice.text


def test_opening_advice():
    tips = " ".join(a.text for a in coach.strategy(chess.Board(), chess.WHITE))
    assert "cavaliers" in tips and "Roquez" in tips and "e4 ou d4" in tips
    black = " ".join(a.text for a in coach.strategy(chess.Board(), chess.BLACK))
    assert "e5 ou d5" in black


def _line(engine, board, **kw):
    return engine.analyse(board, time_limit=0.3, **kw)


def test_action_points_to_free_material(engine):
    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")  # dame noire en prise
    lines = _line(engine, board, multipv=2).lines
    advice = coach.action(board, lines, None)
    assert advice.kind == "action" and "dame noire en d5" in advice.text and "gagner" in advice.text


def test_poisoned_capture_is_flagged():
    board = chess.Board("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1")  # d5 défendu par c6
    advice = coach.poisoned(board, ())
    assert advice is not None and "votre dame" in advice.text and "défendu" in advice.text


def test_blunder_explains_the_refutation(engine):
    before = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    best = _line(engine, before).best
    played = chess.Move.from_uci("d1h5")
    after = before.copy()
    after.push(played)
    reply = _line(engine, after).best
    loss = coach._cp(best, chess.WHITE) - coach._cp(reply, chess.WHITE)
    text = coach.move_quality(max(loss, 150), best, before, played, reply).text
    assert "l'IA peut répondre" in text and "Il fallait" in text


def test_summary_in_words(engine):
    board = chess.Board("4k3/8/8/8/8/8/8/QQ2K3 w - - 0 1")
    words, level = coach.summary(board, _line(engine, board).best, chess.WHITE)
    assert level == "bien" and ("gagnant" in words or "mat" in words)


def test_advise_puts_one_priority(engine):
    board = chess.Board()
    board.push_san("e4")
    analysis = _line(engine, board, multipv=3)
    advices = coach.advise(board, analysis.lines, None)
    kinds = [a.kind for a in advices]
    assert kinds.count("action") == 1 and "adversaire" in kinds

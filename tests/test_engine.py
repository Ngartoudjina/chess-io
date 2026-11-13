import chess
import pytest

from chessio.engine import Engine, EngineNotFoundError, find_stockfish, format_score
from chessio.position import InvalidPositionError


@pytest.fixture(scope="module")
def engine():
    try:
        path = find_stockfish()
    except EngineNotFoundError:
        pytest.skip("Stockfish absent : lancer scripts/fetch_stockfish.sh")
    with Engine(path, threads=1, hash_mb=16) as eng:
        yield eng


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (chess.engine.PovScore(chess.engine.Cp(35), chess.WHITE), "+0.35"),
        (chess.engine.PovScore(chess.engine.Cp(35), chess.BLACK), "-0.35"),
        (chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE), "#3"),
        (chess.engine.PovScore(chess.engine.Mate(2), chess.BLACK), "#-2"),
    ],
)
def test_format_score_is_white_pov(score, expected):
    assert format_score(score) == expected


def test_finds_back_rank_mate(engine):
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    best = engine.analyse(board, time_limit=0.5).best
    assert best is not None
    assert best.uci == "a1a8"
    assert best.san == "Ra8#"
    assert best.evaluation == "#1"


def test_finds_mate_for_black(engine):
    board = chess.Board("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1")
    best = engine.analyse(board, time_limit=0.5).best
    assert best.uci == "a8a1"
    assert best.evaluation == "#-1"


def test_multipv_returns_distinct_moves(engine):
    analysis = engine.analyse(chess.Board(), time_limit=0.5, multipv=3)
    moves = [line.move for line in analysis.lines]
    assert len(moves) == 3
    assert len(set(moves)) == 3
    assert all(m in chess.Board().legal_moves for m in moves)


def test_respects_time_limit(engine):
    analysis = engine.analyse(chess.Board(), time_limit=0.3)
    assert analysis.elapsed < 1.5


def test_refuses_invalid_position(engine):
    with pytest.raises(InvalidPositionError):
        engine.analyse(chess.Board("4k3/8/8/8/8/8/8/4RK2 w - - 0 1"), time_limit=0.1)


def test_game_already_over(engine):
    mated = chess.Board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")
    analysis = engine.analyse(mated, time_limit=0.1)
    assert analysis.outcome == "mat"
    assert analysis.best is None


@pytest.mark.parametrize("seconds", [0.05, 10.5])
def test_time_limit_bounds(engine, seconds):
    with pytest.raises(ValueError):
        engine.analyse(chess.Board(), time_limit=seconds)


def test_progressive_results(engine):
    updates = []
    final = engine.analyse(chess.Board(), time_limit=1.0, multipv=2, on_update=updates.append)
    assert updates, "aucun résultat provisoire"
    assert all(update.partial for update in updates)
    assert not final.partial and not final.stopped
    assert final.best.depth >= updates[0].best.depth


def test_stop_interrupts_quickly(engine):
    updates = []
    final = engine.analyse(
        chess.Board(),
        time_limit=5.0,
        on_update=updates.append,
        should_stop=lambda: bool(updates),
    )
    assert final.stopped
    assert final.elapsed < 1.5
    assert final.best is not None


def test_win_draw_loss(engine):
    best = engine.analyse(chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"), time_limit=0.3).best
    assert sum(best.white_wdl) == 1000
    assert best.white_expectation > 0.99
    start = engine.analyse(chess.Board(), time_limit=0.3).best
    assert 0.3 < start.white_expectation < 0.8

"""Les positions de recette L1 doivent être légales : sinon la recette mesurerait du bruit."""

import importlib.util
import sys
from pathlib import Path

import pytest

from chessio.position import validate

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluer_captures.py"
spec = importlib.util.spec_from_file_location("evaluer_captures", SCRIPT)
evaluer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluer  # requis par dataclasses avant l'exécution
spec.loader.exec_module(evaluer)


@pytest.mark.parametrize("sample", evaluer.SAMPLES, ids=lambda s: s.filename)
def test_sample_is_valid(sample):
    assert validate(sample.expected) == []


def test_twenty_distinct_samples():
    assert len({s.expected.board_fen() for s in evaluer.SAMPLES}) == 20
    assert [s.number for s in evaluer.SAMPLES] == list(range(1, 21))

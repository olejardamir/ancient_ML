from pathlib import Path

from ancient_ml.data import load_draws


def test_load_example_draws():
    draws = load_draws(Path(__file__).parent.parent / "data" / "draws_example.csv")
    assert len(draws) == 7
    assert draws[0].draw_id == 4421
    assert draws[0].main_numbers == (3, 5, 6, 28, 30, 37)
    assert draws[0].bonus == 19

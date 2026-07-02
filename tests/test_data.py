from pathlib import Path

from ancient_ml.data import load_draws, summarize_draws


def test_load_example_draws():
    draws = load_draws(Path(__file__).parent.parent / "data" / "draws_example.csv")
    assert len(draws) == 7
    assert draws[0].draw_id == 4421
    assert draws[0].main_numbers == (3, 5, 6, 28, 30, 37)
    assert draws[0].bonus == 19


def test_load_official_header_format(tmp_path):
    p = tmp_path / "649.csv"
    p.write_text(
        '"PRODUCT","DRAW NUMBER","SEQUENCE NUMBER","DRAW DATE","NUMBER DRAWN 1","NUMBER DRAWN 2","NUMBER DRAWN 3","NUMBER DRAWN 4","NUMBER DRAWN 5","NUMBER DRAWN 6","BONUS NUMBER"\n'
        '"649",1,0,"1982-06-12",3,11,12,14,41,43,13\n',
        encoding="utf-8",
    )
    draws = load_draws(p)
    assert len(draws) == 1
    assert draws[0].draw_id == 1
    assert draws[0].draw_date.isoformat() == "1982-06-12"
    assert draws[0].main_numbers == (3, 11, 12, 14, 41, 43)
    assert draws[0].bonus == 13


def test_summarize_draws():
    draws = load_draws(Path(__file__).parent.parent / "data" / "draws_example.csv")
    summary = summarize_draws(draws)
    assert summary["count"] == 7
    assert summary["first_date"] == "2026-06-03"
    assert summary["last_date"] == "2026-06-24"

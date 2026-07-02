from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import csv
from pathlib import Path
from typing import Iterable


OFFICIAL_649_HEADER = [
    "PRODUCT",
    "DRAW NUMBER",
    "SEQUENCE NUMBER",
    "DRAW DATE",
    "NUMBER DRAWN 1",
    "NUMBER DRAWN 2",
    "NUMBER DRAWN 3",
    "NUMBER DRAWN 4",
    "NUMBER DRAWN 5",
    "NUMBER DRAWN 6",
    "BONUS NUMBER",
]


@dataclass(frozen=True)
class Draw:
    game: str
    draw_id: int
    draw_date: date
    main_numbers: tuple[int, int, int, int, int, int]
    bonus: int

    @property
    def main_set(self) -> set[int]:
        return set(self.main_numbers)


def _clean_cell(value: str) -> str:
    return value.strip().strip('"').strip()


def is_header_row(row: list[str]) -> bool:
    """Return true for the official Lotto 6/49 CSV header.

    The project accepts both headerless rows pasted from earlier experiments and
    the official `649.csv` format with column names:

        PRODUCT,DRAW NUMBER,SEQUENCE NUMBER,DRAW DATE,...,BONUS NUMBER
    """
    if not row:
        return False
    cleaned = [_clean_cell(c).upper() for c in row]
    return cleaned[: len(OFFICIAL_649_HEADER)] == OFFICIAL_649_HEADER


def parse_draw_row(row: list[str]) -> Draw:
    """Parse one Lotto 6/49 row.

    Accepted input formats:

    1. Official CSV row with or without header:
       PRODUCT,DRAW NUMBER,SEQUENCE NUMBER,DRAW DATE,NUMBER DRAWN 1..6,BONUS NUMBER

    2. Headerless row in the same positional format:
       game, draw_id, ignored, yyyy-mm-dd, n1, n2, n3, n4, n5, n6, bonus
    """
    if len(row) < 11:
        raise ValueError(f"Expected at least 11 columns, got {len(row)}: {row!r}")

    game = _clean_cell(row[0])
    draw_id = int(_clean_cell(row[1]))
    draw_date = date.fromisoformat(_clean_cell(row[3]))
    nums = tuple(int(_clean_cell(x)) for x in row[4:10])
    if len(nums) != 6:
        raise ValueError(f"Expected 6 main numbers: {row!r}")
    for n in nums:
        if not 1 <= n <= 49:
            raise ValueError(f"Main number outside 1..49: {n}")
    bonus = int(_clean_cell(row[10]))
    if not 1 <= bonus <= 49:
        raise ValueError(f"Bonus outside 1..49: {bonus}")
    return Draw(game=game, draw_id=draw_id, draw_date=draw_date, main_numbers=nums, bonus=bonus)


def load_draws(path: str | Path, *, product: str = "649") -> list[Draw]:
    """Load Lotto 6/49 draws from `path`.

    The loader is intentionally strict about number ranges but tolerant of:
    - UTF-8 BOMs
    - quoted CSV cells
    - the official header row
    - extra whitespace

    By default it keeps only PRODUCT == "649" rows.
    """
    draws: list[Draw] = []
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, start=1):
            if not row or all(not c.strip() for c in row):
                continue
            if is_header_row(row):
                continue
            try:
                draw = parse_draw_row(row)
            except Exception as exc:
                raise ValueError(f"Invalid draw row {i} in {path}: {row!r}: {exc}") from exc
            if product and draw.game != product:
                continue
            draws.append(draw)
    draws.sort(key=lambda d: (d.draw_date, d.draw_id))
    return draws


def summarize_draws(draws: list[Draw]) -> dict[str, object]:
    if not draws:
        return {
            "count": 0,
            "first_date": None,
            "last_date": None,
            "first_draw_id": None,
            "last_draw_id": None,
        }
    return {
        "count": len(draws),
        "first_date": draws[0].draw_date.isoformat(),
        "last_date": draws[-1].draw_date.isoformat(),
        "first_draw_id": draws[0].draw_id,
        "last_draw_id": draws[-1].draw_id,
    }


def split_time_ordered(draws: list[Draw], train_fraction: float = 0.70) -> tuple[list[Draw], list[Draw]]:
    if not draws:
        return [], []
    if len(draws) < 3:
        return draws, draws
    cut = max(1, min(len(draws) - 1, int(len(draws) * train_fraction)))
    return draws[:cut], draws[cut:]


def split_time_ordered_3(
    draws: list[Draw],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[list[Draw], list[Draw], list[Draw]]:
    if not draws:
        return [], [], []
    n = len(draws)
    train_cut = max(1, min(n - 2, int(n * train_fraction)))
    val_cut = max(train_cut + 1, min(n - 1, int(n * (train_fraction + validation_fraction))))
    return draws[:train_cut], draws[train_cut:val_cut], draws[val_cut:]

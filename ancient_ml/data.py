from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import csv
from pathlib import Path
from typing import Iterable, List


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


def parse_draw_row(row: list[str]) -> Draw:
    """Parse a Lotto 6/49 row.

    Expected input:
        game, draw_id, ignored, yyyy-mm-dd, n1, n2, n3, n4, n5, n6, bonus
    """
    if len(row) < 11:
        raise ValueError(f"Expected at least 11 columns, got {len(row)}: {row!r}")
    game = row[0].strip().strip('"')
    draw_id = int(row[1])
    draw_date = date.fromisoformat(row[3].strip().strip('"'))
    nums = tuple(int(x) for x in row[4:10])
    if len(nums) != 6:
        raise ValueError(f"Expected 6 main numbers: {row!r}")
    for n in nums:
        if not 1 <= n <= 49:
            raise ValueError(f"Main number outside 1..49: {n}")
    bonus = int(row[10])
    if not 1 <= bonus <= 49:
        raise ValueError(f"Bonus outside 1..49: {bonus}")
    return Draw(game=game, draw_id=draw_id, draw_date=draw_date, main_numbers=nums, bonus=bonus)


def load_draws(path: str | Path) -> list[Draw]:
    draws: list[Draw] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, start=1):
            if not row or all(not c.strip() for c in row):
                continue
            try:
                draws.append(parse_draw_row(row))
            except Exception as exc:
                raise ValueError(f"Invalid draw row {i}: {row!r}: {exc}") from exc
    draws.sort(key=lambda d: d.draw_date)
    return draws


def split_time_ordered(draws: list[Draw], train_fraction: float = 0.70) -> tuple[list[Draw], list[Draw]]:
    if not draws:
        return [], []
    if len(draws) < 3:
        return draws, draws
    cut = max(1, min(len(draws) - 1, int(len(draws) * train_fraction)))
    return draws[:cut], draws[cut:]

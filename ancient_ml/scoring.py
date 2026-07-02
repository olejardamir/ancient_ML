from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .data import Draw


@dataclass(frozen=True)
class Prediction:
    main_numbers: tuple[int, int, int, int, int, int]
    bonus: int

    @classmethod
    def from_lists(cls, main: list[int], bonus: int) -> "Prediction":
        if len(main) != 6:
            raise ValueError("Prediction must contain 6 main numbers")
        return cls(tuple(sorted(int(n) for n in main)), int(bonus))


@dataclass(frozen=True)
class Score:
    main_hits: int
    bonus_in_main: bool
    main_hit_bonus: bool
    points: float


def score_prediction(pred: Prediction, draw: Draw) -> Score:
    pred_main = set(pred.main_numbers)
    main_hits = len(pred_main & draw.main_set)
    bonus_in_main = pred.bonus in draw.main_set
    main_hit_bonus = draw.bonus in pred_main
    points = float(main_hits)
    if bonus_in_main:
        points += 0.25
    if main_hit_bonus:
        points += 0.50
    if main_hits >= 3:
        points += 2.0
    return Score(main_hits, bonus_in_main, main_hit_bonus, points)


def score_predictions(preds: list[Prediction], draws: list[Draw]) -> dict:
    if len(preds) != len(draws):
        raise ValueError("preds and draws must have same length")
    scores = [score_prediction(p, d) for p, d in zip(preds, draws)]
    hist: dict[int, int] = {}
    for s in scores:
        hist[s.main_hits] = hist.get(s.main_hits, 0) + 1
    total_slots = 6 * len(draws)
    return {
        "draws": len(draws),
        "main_hits": sum(s.main_hits for s in scores),
        "total_slots": total_slots,
        "hits_per_draw": (sum(s.main_hits for s in scores) / len(draws)) if draws else 0.0,
        "points": sum(s.points for s in scores),
        "bonus_in_main": sum(1 for s in scores if s.bonus_in_main),
        "main_hit_bonus": sum(1 for s in scores if s.main_hit_bonus),
        "hit_distribution": hist,
    }


def random_expected_hits_per_draw() -> float:
    return 36.0 / 49.0

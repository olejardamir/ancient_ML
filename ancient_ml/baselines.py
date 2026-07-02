"""Frequency-based baselines for Lotto 6/49 prediction evaluation.

A model should only be promoted if it beats these on validation/test.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from statistics import mean
from typing import Iterable

from .data import Draw
from .scoring import Prediction, score_predictions


def random_prediction() -> Prediction:
    """Uniform random baseline: no skill."""
    import random as _random
    nums = sorted(_random.sample(range(1, 50), 6))
    remaining = [n for n in range(1, 50) if n not in nums]
    bonus = _random.choice(remaining)
    return Prediction.from_lists(nums, bonus)


def random_baseline(draws: list[Draw]) -> dict:
    """Expected performance of random guessing over the given draws."""
    import random as _random
    rng = _random.Random(42)
    preds = [random_prediction() for _ in draws]
    rng.shuffle(preds)
    return score_predictions(preds, draws)


def _global_frequencies(draws: list[Draw]) -> dict[int, float]:
    """Global frequency of each number across all draws."""
    counter: Counter[int] = Counter()
    for d in draws:
        counter.update(d.main_numbers)
    total = sum(counter.values())
    if total == 0:
        return {n: 0.0 for n in range(1, 50)}
    return {n: counter[n] / total for n in range(1, 50)}


def _top_6_from_freq(freqs: dict[int, float]) -> list[int]:
    sorted_nums = sorted(freqs.items(), key=lambda kv: (-kv[1], kv[0]))
    return [n for n, _ in sorted_nums[:6]]


def _bonus_from_freq(freqs: dict[int, float], exclude: set[int]) -> int:
    sorted_nums = sorted(freqs.items(), key=lambda kv: (-kv[1], kv[0]))
    for n, _ in sorted_nums:
        if n not in exclude:
            return n
    return 1


def frequency_baseline(draws: list[Draw], train_draws: list[Draw]) -> dict:
    """Global frequency baseline: pick top-6 most frequent numbers from training."""
    freqs = _global_frequencies(train_draws)
    main = _top_6_from_freq(freqs)
    bonus = _bonus_from_freq(freqs, set(main))
    pred = Prediction.from_lists(main, bonus)
    return score_predictions([pred] * len(draws), draws)


def _rolling_frequencies(draws: list[Draw], window_days: int, reference_date: date) -> dict[int, float]:
    """Frequency of each number within a rolling window before reference_date."""
    cutoff = reference_date - timedelta(days=window_days)
    recent = [d for d in draws if d.draw_date >= cutoff]
    return _global_frequencies(recent)


def rolling_frequency_baseline(draws: list[Draw], train_draws: list[Draw], window_days: int = 90) -> dict:
    """Rolling window frequency baseline."""
    if not draws:
        return score_predictions([], draws)
    ref = draws[0].draw_date
    freqs = _rolling_frequencies(train_draws, window_days, ref)
    main = _top_6_from_freq(freqs)
    bonus = _bonus_from_freq(freqs, set(main))
    pred = Prediction.from_lists(main, bonus)
    return score_predictions([pred] * len(draws), draws)


def repeat_last_draw_baseline(draws: list[Draw], train_draws: list[Draw]) -> dict:
    """Repeat the last draw's main numbers as prediction."""
    if not train_draws or not draws:
        return score_predictions([], draws)
    last = train_draws[-1]
    pred = Prediction.from_lists(list(last.main_numbers), last.bonus)
    return score_predictions([pred] * len(draws), draws)


def all_baselines(train_draws: list[Draw], test_draws: list[Draw]) -> dict:
    """Compute all baselines and return a summary dict."""
    return {
        "random": random_baseline(test_draws),
        "global_frequency": frequency_baseline(test_draws, train_draws),
        "rolling_90d_frequency": rolling_frequency_baseline(test_draws, train_draws, window_days=90),
        "rolling_365d_frequency": rolling_frequency_baseline(test_draws, train_draws, window_days=365),
        "repeat_last_draw": repeat_last_draw_baseline(test_draws, train_draws),
    }

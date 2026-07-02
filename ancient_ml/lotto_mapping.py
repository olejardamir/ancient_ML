"""Experimental Lotto 1..49 mapping functions.

This is the experimental layer that converts Vedic chart factors into the 1..49
domain. The chart computations in vedic_engine.py are traditional Jyotish; the
folding and ranking here is necessarily experimental.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from math import cos, pi, sin

from .vedic_engine import (
    PLANETS,
    SyntheticSeed,
    build_kundli,
    house_from_lagna,
    transit_positions,
)


def fold_1_49(x: float | int) -> int:
    """Map any numeric chart factor into the Lotto 1..49 domain."""
    return int(abs(round(float(x)))) % 49 + 1


def circular_distance_49(a: int, b: int) -> float:
    d = abs(a - b)
    return float(min(d, 49 - d)) / 24.5


@lru_cache(maxsize=4096)
def _cached_kundli(seed_tuple: tuple) -> object:
    seed = SyntheticSeed(*seed_tuple)
    return build_kundli(seed)


def _seed_to_tuple(seed: SyntheticSeed) -> tuple:
    return (
        seed.year,
        seed.month,
        seed.day,
        seed.hour,
        seed.minute,
        seed.second,
        float(seed.latitude),
        float(seed.longitude),
        seed.ayanamsha,
    )


def candidate_anchor_numbers(seed: SyntheticSeed, draw_date: date) -> list[int]:
    """Cheap deterministic Vedic-derived anchor numbers for a seed/date.

    The chart computations are traditional; folding chart factors into 1..49 is the
    experimental Lotto layer.
    """
    kundli = _cached_kundli(_seed_to_tuple(seed))
    transits = transit_positions(draw_date, seed.ayanamsha, hour_utc=0.0)
    anchors: list[int] = []

    anchors.append(fold_1_49(kundli.lagna_longitude))
    anchors.append(fold_1_49((kundli.lagna_sign_index + 1) * 4))

    for planet_name in PLANETS:
        g = kundli.grahas[planet_name]
        t = transits[planet_name]
        anchors.extend(
            [
                fold_1_49(g.longitude),
                fold_1_49(g.degree_in_sign),
                fold_1_49((g.nakshatra_index + 1) * g.pada),
                fold_1_49((g.sign_index + 1) * (g.navamsa_sign_index + 1)),
                fold_1_49(abs(t.longitude - g.longitude)),
                fold_1_49((t.nakshatra_index + 1) * t.pada),
            ]
        )
        house = house_from_lagna(kundli.lagna_sign_index, g.sign_index)
        anchors.append(fold_1_49(house * (g.pada + 1)))

    return anchors


def cheap_rank_numbers(seed: SyntheticSeed, draw_date: date) -> list[tuple[int, float]]:
    anchors = candidate_anchor_numbers(seed, draw_date)
    counts = {n: 0 for n in range(1, 50)}
    for a in anchors:
        counts[a] += 1
        # small spillover to neighbors to avoid brittle exactness
        counts[(a % 49) + 1] += 0.20
        counts[((a - 2) % 49) + 1] += 0.20

    # Traditional-factor-inspired stable weights: do not dominate, only break ties.
    for n in range(1, 50):
        counts[n] += 0.015 * (sin(2 * pi * n / 27.0) + cos(2 * pi * n / 12.0))

    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def cheap_predict(seed: SyntheticSeed, draw_date: date) -> tuple[list[int], int]:
    ranked = cheap_rank_numbers(seed, draw_date)
    main = sorted(n for n, _ in ranked[:6])
    bonus = next(n for n, _ in ranked[6:] if n not in main)
    return main, bonus

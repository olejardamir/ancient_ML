from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from math import cos, pi, sin
from typing import Iterable

import numpy as np

from .vedic_engine import (
    DASHA_ORDER,
    PLANETS,
    SIGNS,
    SyntheticSeed,
    build_kundli,
    house_from_lagna,
    transit_positions,
    vimshottari_lords,
)


@dataclass(frozen=True)
class FeatureConfig:
    include_transits: bool = True
    include_dasha: bool = True
    include_number_harmonics: bool = True


def fold_1_49(x: float | int) -> int:
    """Map any numeric chart factor into the Lotto 1..49 domain.

    This is the experimental layer. The input factors themselves are produced from
    traditional Jyotish calculations.
    """
    return int(abs(round(float(x)))) % 49 + 1


def circular_distance_49(a: int, b: int) -> float:
    d = abs(a - b)
    return float(min(d, 49 - d)) / 24.5


@lru_cache(maxsize=4096)
def _cached_kundli(seed_tuple: tuple) -> object:
    seed = SyntheticSeed(*seed_tuple)
    return build_kundli(seed)


def seed_to_tuple(seed: SyntheticSeed) -> tuple:
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
    kundli = _cached_kundli(seed_to_tuple(seed))
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


def feature_vector(seed: SyntheticSeed, draw_date: date, number: int, cfg: FeatureConfig | None = None) -> np.ndarray:
    cfg = cfg or FeatureConfig()
    kundli = _cached_kundli(seed_to_tuple(seed))
    transits = transit_positions(draw_date, seed.ayanamsha, hour_utc=0.0)
    feats: list[float] = []

    # Number harmonics help the model represent cyclical folded Jyotish factors.
    if cfg.include_number_harmonics:
        feats.extend([
            number / 49.0,
            sin(2 * pi * number / 49.0),
            cos(2 * pi * number / 49.0),
            sin(2 * pi * number / 27.0),
            cos(2 * pi * number / 27.0),
            sin(2 * pi * number / 12.0),
            cos(2 * pi * number / 12.0),
        ])

    # Lagna features
    lagna_anchor = fold_1_49(kundli.lagna_longitude)
    feats.extend([
        1.0 - circular_distance_49(number, lagna_anchor),
        1.0 if number == fold_1_49((kundli.lagna_sign_index + 1) * 4) else 0.0,
    ])

    for planet_name in PLANETS:
        g = kundli.grahas[planet_name]
        anchors = [
            fold_1_49(g.longitude),
            fold_1_49(g.degree_in_sign),
            fold_1_49((g.nakshatra_index + 1) * g.pada),
            fold_1_49((g.sign_index + 1) * (g.navamsa_sign_index + 1)),
            fold_1_49(house_from_lagna(kundli.lagna_sign_index, g.sign_index) * (g.pada + 1)),
        ]
        feats.extend([1.0 - circular_distance_49(number, a) for a in anchors])
        feats.append(float(number in anchors))

        if cfg.include_transits:
            t = transits[planet_name]
            trans_anchor = [
                fold_1_49(t.longitude),
                fold_1_49(t.degree_in_sign),
                fold_1_49((t.nakshatra_index + 1) * t.pada),
                fold_1_49(abs(t.longitude - g.longitude)),
            ]
            feats.extend([1.0 - circular_distance_49(number, a) for a in trans_anchor])
            feats.append(float(number in trans_anchor))

    if cfg.include_dasha:
        maha, antar = vimshottari_lords(kundli.grahas["Moon"].longitude, kundli.jd_ut, kundli.jd_ut)
        # Synthetic draw-time dasha from natal Moon, using draw date at 00:00 UT.
        # Import locally to avoid cycle on module import in some test contexts.
        from .vedic_engine import julian_day_from_date
        draw_jd = julian_day_from_date(draw_date, hour_utc=0.0)
        maha_draw, antar_draw = vimshottari_lords(kundli.grahas["Moon"].longitude, kundli.jd_ut, draw_jd)
        for lord in DASHA_ORDER:
            feats.append(float(maha_draw == lord))
            feats.append(float(antar_draw == lord))
        # Lord-specific anchors
        for lord in (maha_draw, antar_draw):
            if lord in kundli.grahas:
                feats.append(1.0 - circular_distance_49(number, fold_1_49(kundli.grahas[lord].longitude)))
            else:
                feats.append(0.0)

    return np.asarray(feats, dtype=np.float32)


def build_training_matrix(seed: SyntheticSeed, draws: Iterable, cfg: FeatureConfig | None = None):
    xs: list[np.ndarray] = []
    y: list[int] = []
    weights: list[float] = []
    for draw in draws:
        main = draw.main_set
        for number in range(1, 50):
            xs.append(feature_vector(seed, draw.draw_date, number, cfg=cfg))
            if number in main:
                y.append(1)
                weights.append(1.0)
            elif number == draw.bonus:
                y.append(0)
                weights.append(0.35)
            else:
                y.append(0)
                weights.append(0.10)
    return np.vstack(xs), np.asarray(y, dtype=np.int8), np.asarray(weights, dtype=np.float32)

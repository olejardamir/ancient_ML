from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from math import cos, pi, sin
from typing import Iterable

import numpy as np

from .lotto_mapping import circular_distance_49, fold_1_49, _cached_kundli, _seed_to_tuple
from .vedic_engine import (
    DASHA_ORDER,
    GrahaPosition,
    PLANETS,
    SyntheticSeed,
    build_kundli,
    house_from_lagna,
    transit_positions,
    vimshottari_lords,
)

_TRANSIT_CACHE: dict[str, dict[str, GrahaPosition]] | None = None


def set_transit_cache(cache: dict[str, dict[str, GrahaPosition]]) -> None:
    global _TRANSIT_CACHE
    _TRANSIT_CACHE = cache


def _get_transits(draw_date_iso: str, ayanamsha: str, hour_utc: float = 0.0) -> dict[str, GrahaPosition]:
    if _TRANSIT_CACHE is not None:
        return _TRANSIT_CACHE[draw_date_iso]
    return transit_positions(date.fromisoformat(draw_date_iso), ayanamsha, hour_utc=hour_utc)


@dataclass(frozen=True)
class FeatureConfig:
    include_transits: bool = True
    include_dasha: bool = True
    include_number_harmonics: bool = True


def feature_vector(seed: SyntheticSeed, draw_date: date, number: int, cfg: FeatureConfig | None = None) -> np.ndarray:
    cfg = cfg or FeatureConfig()
    kundli = _cached_kundli(_seed_to_tuple(seed))
    transits = _get_transits(draw_date.isoformat(), seed.ayanamsha, 0.0)
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
                weights.append(0.50)
            else:
                y.append(0)
                weights.append(0.10)
    return np.vstack(xs), np.asarray(y, dtype=np.int8), np.asarray(weights, dtype=np.float32)

"""Configuration loading for the ancient_ML framework."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CheapSearchConfig:
    train_fraction: float = 0.70
    validation_weight: float = 2.0
    bonus_main_weight: float = 0.25
    main_bonus_weight: float = 0.50
    three_hit_bonus: float = 2.0
    repeat_penalty: float = 0.04


@dataclass
class SeedSpaceConfig:
    year_min: int = -5000
    year_max: int = 5000
    latitude_min: float = -60.0
    latitude_max: float = 60.0
    longitude_min: float = -180.0
    longitude_max: float = 180.0


@dataclass
class Config:
    ayanamsha: str = "LAHIRI"
    cheap_search: CheapSearchConfig = field(default_factory=CheapSearchConfig)
    seed_space: SeedSpaceConfig = field(default_factory=SeedSpaceConfig)


def load_config(path: str | Path | None) -> Config:
    if path is None:
        return Config()
    p = Path(path)
    if not p.exists():
        return Config()
    raw: dict[str, Any] = json.loads(p.read_text())
    cfg = Config()
    if "ayanamsha" in raw:
        cfg.ayanamsha = raw["ayanamsha"]
    if "cheap_search" in raw:
        cs = raw["cheap_search"]
        for key in ("train_fraction", "validation_weight", "bonus_main_weight", "main_bonus_weight", "three_hit_bonus", "repeat_penalty"):
            if key in cs:
                setattr(cfg.cheap_search, key, cs[key])
    if "seed_space" in raw:
        ss = raw["seed_space"]
        for key in ("year_min", "year_max", "latitude_min", "latitude_max", "longitude_min", "longitude_max"):
            if key in ss:
                setattr(cfg.seed_space, key, ss[key])
    return cfg

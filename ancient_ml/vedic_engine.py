from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from math import floor
from typing import Dict, Iterable

try:
    import swisseph as swe
except Exception:  # pragma: no cover - runtime dependency error is clearer later
    swe = None  # type: ignore


NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni",
    "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha",
    "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana",
    "Dhanishta", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati",
]

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
DASHA_ORDER = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
DASHA_YEARS = {
    "Ketu": 7,
    "Venus": 20,
    "Sun": 6,
    "Moon": 10,
    "Mars": 7,
    "Rahu": 18,
    "Jupiter": 16,
    "Saturn": 19,
    "Mercury": 17,
}


@dataclass(frozen=True)
class SyntheticSeed:
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    latitude: float
    longitude: float
    ayanamsha: str = "LAHIRI"

    def iso_datetime(self) -> str:
        y = f"{self.year:+06d}" if self.year <= 0 or self.year > 9999 else f"{self.year:04d}"
        return f"{y}-{self.month:02d}-{self.day:02d}T{self.hour:02d}:{self.minute:02d}:{self.second:02d}Z"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SyntheticSeed":
        return cls(**d)


@dataclass(frozen=True)
class GrahaPosition:
    name: str
    longitude: float
    sign_index: int
    sign_name: str
    degree_in_sign: float
    nakshatra_index: int
    nakshatra_name: str
    pada: int
    navamsa_sign_index: int
    navamsa_sign_name: str


@dataclass(frozen=True)
class Kundli:
    seed: SyntheticSeed
    jd_ut: float
    lagna_longitude: float
    lagna_sign_index: int
    lagna_sign_name: str
    grahas: dict[str, GrahaPosition]
    vimshottari_mahadasha: str
    vimshottari_antardasha: str


def require_swe() -> None:
    if swe is None:
        raise RuntimeError(
            "pyswisseph is required. Install with: pip install pyswisseph"
        )


def _set_ayanamsha(name: str) -> None:
    require_swe()
    name = name.upper()
    if name == "LAHIRI":
        swe.set_sid_mode(swe.SIDM_LAHIRI)
    else:
        raise ValueError(f"Unsupported ayanamsha: {name}. Currently supported: LAHIRI")


def julian_day_from_seed(seed: SyntheticSeed) -> float:
    require_swe()
    hour_float = seed.hour + seed.minute / 60.0 + seed.second / 3600.0
    return swe.julday(seed.year, seed.month, seed.day, hour_float, swe.GREG_CAL)


def julian_day_from_date(d: date, hour_utc: float = 0.0) -> float:
    require_swe()
    return swe.julday(d.year, d.month, d.day, hour_utc, swe.GREG_CAL)


def normalize360(x: float) -> float:
    return x % 360.0


def sign_index(longitude: float) -> int:
    return int(normalize360(longitude) // 30)


def nakshatra_index(longitude: float) -> int:
    return int(normalize360(longitude) // (360.0 / 27.0))


def pada(longitude: float) -> int:
    part = normalize360(longitude) % (360.0 / 27.0)
    return int(part // (360.0 / 108.0)) + 1


def navamsa_sign_index(longitude: float) -> int:
    # Traditional D9: each sign has 9 parts of 3°20'.
    # Movable signs start from same sign, fixed from 9th, dual from 5th.
    s = sign_index(longitude)
    part = int((normalize360(longitude) % 30.0) // (30.0 / 9.0))
    if s in (0, 3, 6, 9):  # movable
        start = s
    elif s in (1, 4, 7, 10):  # fixed
        start = (s + 8) % 12
    else:  # dual
        start = (s + 4) % 12
    return (start + part) % 12


def graha_position(name: str, longitude: float) -> GrahaPosition:
    lon = normalize360(longitude)
    s = sign_index(lon)
    n = nakshatra_index(lon)
    nav = navamsa_sign_index(lon)
    return GrahaPosition(
        name=name,
        longitude=lon,
        sign_index=s,
        sign_name=SIGNS[s],
        degree_in_sign=lon % 30.0,
        nakshatra_index=n,
        nakshatra_name=NAKSHATRAS[n],
        pada=pada(lon),
        navamsa_sign_index=nav,
        navamsa_sign_name=SIGNS[nav],
    )


def _calc_sidereal_body(jd_ut: float, body: int) -> float:
    require_swe()
    flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL
    try:
        result, _flags = swe.calc_ut(jd_ut, body, flags)
    except Exception:
        flags = swe.FLG_MOSEPH | swe.FLG_SIDEREAL
        result, _flags = swe.calc_ut(jd_ut, body, flags)
    return float(result[0])


def _sidereal_lagna(jd_ut: float, latitude: float, longitude: float) -> float:
    require_swe()
    # Equal-house Ascendant. No Placidus fallback needed — equal houses work
    # reliably across the supported latitude range.
    _cusps, ascmc = swe.houses_ex(jd_ut, max(min(latitude, 60.0), -60.0), longitude, b"E")
    tropical_asc = float(ascmc[0])
    ayan = float(swe.get_ayanamsa_ut(jd_ut))
    return normalize360(tropical_asc - ayan)


def build_kundli(seed: SyntheticSeed) -> Kundli:
    _set_ayanamsha(seed.ayanamsha)
    jd = julian_day_from_seed(seed)

    body_ids = {
        "Sun": swe.SUN,
        "Moon": swe.MOON,
        "Mars": swe.MARS,
        "Mercury": swe.MERCURY,
        "Jupiter": swe.JUPITER,
        "Venus": swe.VENUS,
        "Saturn": swe.SATURN,
        "Rahu": swe.MEAN_NODE,
    }
    longs = {name: _calc_sidereal_body(jd, body) for name, body in body_ids.items()}
    longs["Ketu"] = normalize360(longs["Rahu"] + 180.0)
    grahas = {name: graha_position(name, lon) for name, lon in longs.items()}
    lagna = _sidereal_lagna(jd, seed.latitude, seed.longitude)
    lagna_s = sign_index(lagna)
    maha, antar = vimshottari_lords(grahas["Moon"].longitude, jd, jd)
    return Kundli(
        seed=seed,
        jd_ut=jd,
        lagna_longitude=lagna,
        lagna_sign_index=lagna_s,
        lagna_sign_name=SIGNS[lagna_s],
        grahas=grahas,
        vimshottari_mahadasha=maha,
        vimshottari_antardasha=antar,
    )


def transit_positions(draw_date: date, ayanamsha: str = "LAHIRI", hour_utc: float = 0.0) -> dict[str, GrahaPosition]:
    _set_ayanamsha(ayanamsha)
    jd = julian_day_from_date(draw_date, hour_utc=hour_utc)
    body_ids = {
        "Sun": swe.SUN,
        "Moon": swe.MOON,
        "Mars": swe.MARS,
        "Mercury": swe.MERCURY,
        "Jupiter": swe.JUPITER,
        "Venus": swe.VENUS,
        "Saturn": swe.SATURN,
        "Rahu": swe.MEAN_NODE,
    }
    longs = {name: _calc_sidereal_body(jd, body) for name, body in body_ids.items()}
    longs["Ketu"] = normalize360(longs["Rahu"] + 180.0)
    return {name: graha_position(name, lon) for name, lon in longs.items()}


def house_from_lagna(lagna_sign: int, body_sign: int) -> int:
    return ((body_sign - lagna_sign) % 12) + 1


def vimshottari_lords(moon_longitude: float, birth_jd: float, target_jd: float) -> tuple[str, str]:
    """Approximate Vimshottari mahadasha/antardasha from natal Moon.

    This uses standard 120-year Vimshottari order and nakshatra balance at birth.
    It is sufficient for feature generation; exact traditional software can be used later
    if sub-sub-period precision is needed.
    """
    nak = nakshatra_index(moon_longitude)
    lord = DASHA_ORDER[nak % 9]
    nak_start = nak * (360.0 / 27.0)
    elapsed_in_nak = normalize360(moon_longitude) - nak_start
    frac_elapsed = elapsed_in_nak / (360.0 / 27.0)
    lord_years = DASHA_YEARS[lord]
    remaining_years = lord_years * (1.0 - frac_elapsed)
    elapsed_years = max(0.0, (target_jd - birth_jd) / 365.2425)

    # Build sequence starting at birth dasha lord with remaining balance first.
    idx = DASHA_ORDER.index(lord)
    if elapsed_years < remaining_years:
        maha = lord
        antar = _antardasha_within(maha, elapsed_years, remaining_years, first_balance=True)
        return maha, antar
    elapsed_years -= remaining_years
    idx = (idx + 1) % 9
    while True:
        maha = DASHA_ORDER[idx]
        yrs = DASHA_YEARS[maha]
        if elapsed_years < yrs:
            return maha, _antardasha_within(maha, elapsed_years, yrs, first_balance=False)
        elapsed_years -= yrs
        idx = (idx + 1) % 9


def _antardasha_within(maha: str, elapsed_years: float, maha_span: float, first_balance: bool) -> str:
    start = DASHA_ORDER.index(maha)
    cursor = 0.0
    for i in range(9):
        lord = DASHA_ORDER[(start + i) % 9]
        span = maha_span * DASHA_YEARS[lord] / 120.0
        if elapsed_years <= cursor + span:
            return lord
        cursor += span
    return DASHA_ORDER[(start + 8) % 9]

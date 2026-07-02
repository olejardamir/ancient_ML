from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
import time
from typing import Any

from .vedic_engine import SyntheticSeed

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    seed_id TEXT PRIMARY KEY,
    seed_json TEXT NOT NULL,
    train_score REAL NOT NULL,
    validation_score REAL NOT NULL,
    total_score REAL NOT NULL,
    hits_per_draw REAL NOT NULL,
    created_at REAL NOT NULL,
    trained INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    seed_id TEXT NOT NULL,
    model_path TEXT NOT NULL,
    train_score REAL NOT NULL,
    validation_score REAL NOT NULL,
    promoted INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_candidates_score ON candidates(validation_score DESC, total_score DESC);
CREATE INDEX IF NOT EXISTS idx_models_promoted ON models(promoted, validation_score DESC);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def seed_id(seed: SyntheticSeed) -> str:
    raw = json.dumps(seed.to_dict(), sort_keys=True)
    import hashlib
    return "kundli_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def add_candidate(conn: sqlite3.Connection, seed: SyntheticSeed, train_score: float, validation_score: float, total_score: float, hits_per_draw: float) -> str:
    sid = seed_id(seed)
    conn.execute(
        """
        INSERT OR REPLACE INTO candidates
        (seed_id, seed_json, train_score, validation_score, total_score, hits_per_draw, created_at, trained)
        VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM candidates WHERE seed_id = ?), ?), COALESCE((SELECT trained FROM candidates WHERE seed_id = ?), 0))
        """,
        (
            sid,
            json.dumps(seed.to_dict(), sort_keys=True),
            float(train_score),
            float(validation_score),
            float(total_score),
            float(hits_per_draw),
            sid,
            time.time(),
            sid,
        ),
    )
    conn.commit()
    return sid


def best_candidates(conn: sqlite3.Connection, limit: int = 20, only_untrained: bool = False) -> list[sqlite3.Row]:
    where = "WHERE trained = 0" if only_untrained else ""
    return list(
        conn.execute(
            f"SELECT * FROM candidates {where} ORDER BY validation_score DESC, total_score DESC LIMIT ?",
            (limit,),
        )
    )


def row_to_seed(row: sqlite3.Row) -> SyntheticSeed:
    return SyntheticSeed.from_dict(json.loads(row["seed_json"]))


def mark_trained(conn: sqlite3.Connection, seed_id_value: str) -> None:
    conn.execute("UPDATE candidates SET trained = 1 WHERE seed_id = ?", (seed_id_value,))
    conn.commit()


def add_model(conn: sqlite3.Connection, model_id: str, seed_id_value: str, model_path: str, train_score: float, validation_score: float, promote: bool) -> None:
    if promote:
        conn.execute("UPDATE models SET promoted = 0 WHERE promoted = 1")
    conn.execute(
        """
        INSERT OR REPLACE INTO models
        (model_id, seed_id, model_path, train_score, validation_score, promoted, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (model_id, seed_id_value, model_path, float(train_score), float(validation_score), 1 if promote else 0, time.time()),
    )
    conn.commit()


def promoted_model(conn: sqlite3.Connection):
    return conn.execute(
        "SELECT * FROM models WHERE promoted = 1 ORDER BY validation_score DESC LIMIT 1"
    ).fetchone()


def best_model_score(conn: sqlite3.Connection) -> float:
    row = promoted_model(conn)
    return float(row["validation_score"]) if row else float("-inf")

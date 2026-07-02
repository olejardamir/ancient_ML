from __future__ import annotations

import argparse
import json
import time as time_module
from pathlib import Path

import joblib
import numpy as np

from .baselines import all_baselines
from .data import load_draws, split_time_ordered
from .feature_builder import feature_vector
from .lotto_mapping import cheap_predict
from .predictor import predict_with_payload
from .registry import (
    best_candidates,
    connect,
    promoted_model,
    row_to_seed,
)
from .scoring import Prediction, score_predictions, random_expected_hits_per_draw
from .vedic_engine import SyntheticSeed


def run_backtest(args: argparse.Namespace) -> None:
    draws = load_draws(args.data)
    test_fraction = 1.0 - args.train_fraction
    train_draws, test_draws = split_time_ordered(draws, args.train_fraction)
    conn = connect(args.state)

    if not test_draws:
        raise SystemExit("Not enough draws for a test set.")

    print(f"Backtest: {len(train_draws)} train, {len(test_draws)} test draws\n")

    # --- Baselines ---
    print("=== Baselines ===")
    baselines = all_baselines(train_draws, test_draws)
    for name, score in baselines.items():
        hits = score.get("hits_per_draw", 0.0)
        pts = score.get("points", 0.0)
        print(f"  {name:30s}  hits/draw={hits:.4f}  points={pts:.2f}")

    # --- Cheap seed (best candidate from registry) ---
    print("\n=== Cheap seed (best candidate) ===")
    rows = best_candidates(conn, limit=1, only_untrained=False)
    cheap_result = None
    if rows:
        seed = row_to_seed(rows[0])
        cheap_preds = [Prediction.from_lists(*cheap_predict(seed, d.draw_date)) for d in test_draws]
        cheap_result = score_predictions(cheap_preds, test_draws)
        print(f"  hits/draw={cheap_result['hits_per_draw']:.4f}  points={cheap_result['points']:.2f}")
    else:
        print("  No candidates found in registry.")

    # --- Promoted ML model ---
    print("\n=== Promoted ML model ===")
    ml_result = None
    model_row = promoted_model(conn)
    if model_row:
        model_path = Path(model_row["model_path"])
        if not model_path.exists():
            resolved = Path(args.models) / model_path.name
            if resolved.exists():
                model_path = resolved
        if model_path.exists():
            payload = joblib.load(str(model_path))
            ml_preds = [predict_with_payload(payload, d.draw_date) for d in test_draws]
            ml_result = score_predictions(ml_preds, test_draws)
            print(f"  model_id={model_row['model_id']}")
            print(f"  hits/draw={ml_result['hits_per_draw']:.4f}  points={ml_result['points']:.2f}")
            print(f"  hit_distribution={dict(sorted(ml_result['hit_distribution'].items()))}")
            print(f"  bonus_in_main={ml_result['bonus_in_main']}/{ml_result['draws']}")
            print(f"  main_hit_bonus={ml_result['main_hit_bonus']}/{ml_result['draws']}")
        else:
            print("  Model file not found.")
    else:
        print("  No promoted model found.")

    # --- Summary ---
    print("\n=== Summary ===")
    random_expected = random_expected_hits_per_draw()
    print(f"  Random expected hits/draw:  {random_expected:.4f}")

    for name, score in baselines.items():
        print(f"  {name:30s}  hits/draw={score['hits_per_draw']:.4f}")

    if cheap_result:
        print(f"  {'cheap_seed':30s}  hits/draw={cheap_result['hits_per_draw']:.4f}")
    if ml_result:
        print(f"  {'promoted_ml':30s}  hits/draw={ml_result['hits_per_draw']:.4f}")

    # Log to registry
    conn.execute(
        """INSERT INTO backtest_runs
           (started_at, train_draws, test_draws,
            random_hits, global_freq_hits, rolling_90d_hits, rolling_365d_hits, repeat_last_hits,
            cheap_seed_hits, promoted_ml_hits, bonus_behavior)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            time_module.time(),
            len(train_draws),
            len(test_draws),
            baselines["random"]["hits_per_draw"],
            baselines["global_frequency"]["hits_per_draw"],
            baselines["rolling_90d_frequency"]["hits_per_draw"],
            baselines["rolling_365d_frequency"]["hits_per_draw"],
            baselines["repeat_last_draw"]["hits_per_draw"],
            cheap_result["hits_per_draw"] if cheap_result else None,
            ml_result["hits_per_draw"] if ml_result else None,
            json.dumps({
                "ml_bonus_in_main": ml_result["bonus_in_main"] if ml_result else None,
                "ml_main_hit_bonus": ml_result["main_hit_bonus"] if ml_result else None,
            }),
        ),
    )
    conn.commit()


def add_backtest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, help="CSV file, for example data/649.csv")
    parser.add_argument("--state", default="state/ancient_ml.sqlite")
    parser.add_argument("--models", default="models")
    parser.add_argument("--train-fraction", type=float, default=0.70)

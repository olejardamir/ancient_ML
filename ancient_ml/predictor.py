from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import joblib
import numpy as np

from .feature_builder import feature_vector
from .lotto_mapping import cheap_predict
from .registry import best_candidates, connect, log_prediction, promoted_model, row_to_seed
from .scoring import Prediction
from .vedic_engine import SyntheticSeed


def predict_with_payload(payload: dict, draw_date: date) -> Prediction:
    scaler = payload["scaler"]
    classifier = payload["classifier"]
    seed = SyntheticSeed.from_dict(payload["seed"])
    nums = list(range(1, 50))
    X = np.vstack([feature_vector(seed, draw_date, n) for n in nums])
    X_scaled = scaler.transform(X)
    if hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(X_scaled)[:, 1]
    else:
        scores = classifier.decision_function(X_scaled)
    ranked = [n for n, _ in sorted(zip(nums, scores), key=lambda kv: (-kv[1], kv[0]))]
    return Prediction.from_lists(sorted(ranked[:6]), ranked[6])


def run_predict(args: argparse.Namespace) -> None:
    d = date.fromisoformat(args.date)
    conn = connect(args.state)
    model_row = promoted_model(conn)
    result: dict
    if model_row:
        model_path = Path(model_row["model_path"])
        if not model_path.exists():
            # Resolve relative to --models directory.
            resolved = Path(args.models) / model_path.name
            if resolved.exists():
                model_path = resolved
            else:
                print(f"Model file {model_path} not found, falling back to cheap seed.")
                model_row = None
        else:
            model_path = str(model_path)
    if model_row:
        payload = joblib.load(str(model_path))
        pred = predict_with_payload(payload, d)
        result = {
            "draw_date": args.date,
            "main_numbers": list(pred.main_numbers),
            "bonus": pred.bonus,
            "mode": "promoted_ml_model",
            "model_id": model_row["model_id"],
            "seed_id": model_row["seed_id"],
            "model_path": str(model_path),
        }
    else:
        rows = best_candidates(conn, limit=1, only_untrained=False)
        if not rows:
            raise SystemExit("No model or seed found. Run search first, then train.")
        seed = row_to_seed(rows[0])
        main, bonus = cheap_predict(seed, d)
        result = {
            "draw_date": args.date,
            "main_numbers": main,
            "bonus": bonus,
            "mode": "cheap_seed_fallback",
            "seed_id": rows[0]["seed_id"],
            "seed": seed.to_dict(),
        }

    log_prediction(conn, args.date, result["mode"], result.get("model_id"), result.get("seed_id"), result["main_numbers"], result["bonus"])
    print(json.dumps(result, indent=2, sort_keys=True))


def add_predict_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--date", required=True, help="Draw date in YYYY-MM-DD format")
    parser.add_argument("--state", default="state/ancient_ml.sqlite")
    parser.add_argument("--models", default="models")

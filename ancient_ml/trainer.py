from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .data import load_draws, split_time_ordered
from .feature_builder import build_training_matrix, feature_vector
from .registry import (
    add_model,
    best_candidates,
    best_model_score,
    connect,
    mark_trained,
    row_to_seed,
)
from .scoring import Prediction, score_predictions


def _predict_with_model(model, seed, draw_date) -> Prediction:
    nums = list(range(1, 50))
    X = np.vstack([feature_vector(seed, draw_date, n) for n in nums])
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
    ranked = [n for n, _ in sorted(zip(nums, scores), key=lambda kv: (-kv[1], kv[0]))]
    return Prediction.from_lists(sorted(ranked[:6]), ranked[6])


def train_for_seed(seed, train_draws, validation_draws, previous_model_path: str | None = None):
    X_train, y_train, w_train = build_training_matrix(seed, train_draws)
    X_val, y_val, w_val = build_training_matrix(seed, validation_draws)

    if previous_model_path and Path(previous_model_path).exists():
        model = joblib.load(previous_model_path)
        # For pipeline-based SGD, repeated fit continues from a warm-start classifier only
        # if configured that way. We keep this simple and refit cleanly for stability.
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=0.0005,
            l1_ratio=0.10,
            max_iter=2000,
            tol=1e-4,
            class_weight="balanced",
            random_state=42,
        ),
    )
    model.fit(X_train, y_train, sgdclassifier__sample_weight=w_train)

    val_scores = model.predict_proba(X_val)[:, 1]
    ap = float(average_precision_score(y_val, val_scores, sample_weight=w_val))

    train_preds = [_predict_with_model(model, seed, d.draw_date) for d in train_draws]
    val_preds = [_predict_with_model(model, seed, d.draw_date) for d in validation_draws]
    train_score = score_predictions(train_preds, train_draws)
    val_score = score_predictions(val_preds, validation_draws)
    return model, {"average_precision": ap, "train": train_score, "validation": val_score}


def run_train(args: argparse.Namespace) -> None:
    draws = load_draws(args.data)
    train_draws, validation_draws = split_time_ordered(draws, args.train_fraction)
    conn = connect(args.state)
    Path(args.models).mkdir(parents=True, exist_ok=True)
    rows = best_candidates(conn, limit=args.max_candidates, only_untrained=args.only_untrained)
    if not rows:
        print("No candidate seeds found. Run the searcher first.")
        return

    current_best = best_model_score(conn)
    for row in rows:
        seed = row_to_seed(row)
        seed_id = row["seed_id"]
        print(f"Training candidate {seed_id}: {seed.iso_datetime()} lat={seed.latitude:.4f} lon={seed.longitude:.4f}")
        try:
            model, metrics = train_for_seed(seed, train_draws, validation_draws)
        except Exception as exc:
            print(f"Training failed for {seed_id}: {exc}")
            mark_trained(conn, seed_id)
            continue

        val_points = float(metrics["validation"]["points"])
        promote = val_points > current_best
        model_id = "model_" + uuid.uuid4().hex[:12]
        model_path = str(Path(args.models) / f"{model_id}.joblib")
        payload = {
            "model": model,
            "seed": seed.to_dict(),
            "seed_id": seed_id,
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(payload, model_path)
        add_model(
            conn,
            model_id=model_id,
            seed_id_value=seed_id,
            model_path=model_path,
            train_score=float(metrics["train"]["points"]),
            validation_score=val_points,
            promote=promote,
        )
        mark_trained(conn, seed_id)
        if promote:
            current_best = val_points
            print(f"PROMOTED {model_id}: validation_points={val_points:.2f} hits/draw={metrics['validation']['hits_per_draw']:.3f}")
        else:
            print(f"Kept as non-promoted {model_id}: validation_points={val_points:.2f}")


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True)
    parser.add_argument("--state", default="state/ancient_ml.sqlite")
    parser.add_argument("--models", default="models")
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--only-untrained", action="store_true")

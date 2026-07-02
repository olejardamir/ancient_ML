from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time as time_module
import uuid

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from .data import load_draws, split_time_ordered_3
from .feature_builder import build_training_matrix, feature_vector
from .registry import (
    add_model,
    best_candidates,
    best_model_score,
    connect,
    log_training_run,
    mark_trained,
    row_to_seed,
    update_model_status,
)
from .scoring import Prediction, score_predictions


def _predict_with_model(scaler, classifier, seed, draw_date) -> Prediction:
    nums = list(range(1, 50))
    X = np.vstack([feature_vector(seed, draw_date, n) for n in nums])
    X_scaled = scaler.transform(X)
    if hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(X_scaled)[:, 1]
    else:
        scores = classifier.decision_function(X_scaled)
    ranked = [n for n, _ in sorted(zip(nums, scores), key=lambda kv: (-kv[1], kv[0]))]
    return Prediction.from_lists(sorted(ranked[:6]), ranked[6])


def train_for_seed(seed, train_draws, validation_draws, previous_model_path: str | None = None):
    X_train, y_train, w_train = build_training_matrix(seed, train_draws)
    X_val, y_val, w_val = build_training_matrix(seed, validation_draws)

    scaler = StandardScaler()
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=0.0005,
        l1_ratio=0.10,
        max_iter=2000,
        tol=1e-4,
        class_weight=None,
        random_state=42,
    )

    if previous_model_path and Path(previous_model_path).exists():
        prev = joblib.load(previous_model_path)
        scaler = prev["scaler"]
        classifier = prev["classifier"]
        # Continue training: refit scaler on new data, then partial_fit classifier.
        scaler.partial_fit(X_train)
        X_scaled = scaler.transform(X_train)
        classifier.partial_fit(X_scaled, y_train, sample_weight=w_train)
    else:
        X_scaled = scaler.fit_transform(X_train)
        classifier.fit(X_scaled, y_train, sample_weight=w_train)

    X_val_scaled = scaler.transform(X_val)
    val_scores = classifier.predict_proba(X_val_scaled)[:, 1]
    ap = float(average_precision_score(y_val, val_scores, sample_weight=w_val))

    train_preds = [_predict_with_model(scaler, classifier, seed, d.draw_date) for d in train_draws]
    val_preds = [_predict_with_model(scaler, classifier, seed, d.draw_date) for d in validation_draws]
    train_score = score_predictions(train_preds, train_draws)
    val_score = score_predictions(val_preds, validation_draws)
    return {"scaler": scaler, "classifier": classifier}, {"average_precision": ap, "train": train_score, "validation": val_score}


def _existing_model_for_seed(conn, seed_id_value: str) -> str | None:
    row = conn.execute(
        "SELECT model_path FROM models WHERE seed_id = ? ORDER BY created_at DESC LIMIT 1",
        (seed_id_value,),
    ).fetchone()
    return row["model_path"] if row else None


def run_train(args: argparse.Namespace) -> None:
    draws = load_draws(args.data)
    train_draws, validation_draws, _test_draws = split_time_ordered_3(draws, args.train_fraction, args.validation_fraction)
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

        existing_path = _existing_model_for_seed(conn, seed_id)
        if existing_path and Path(existing_path).exists():
            print(f"  Continuing from previous model: {existing_path}")
        else:
            existing_path = None

        t0 = time_module.time()
        try:
            components, metrics = train_for_seed(seed, train_draws, validation_draws, previous_model_path=existing_path)
        except Exception as exc:
            print(f"Training failed for {seed_id}: {exc}")
            mark_trained(conn, seed_id)
            log_training_run(conn, seed_id, None, None, None, None, False, time_module.time() - t0)
            continue

        train_seconds = time_module.time() - t0
        val_points = float(metrics["validation"]["points"])
        ap = float(metrics["average_precision"])

        cheap_seed_val_points = float(row["validation_score"])
        min_improvement = 0.01 * len(validation_draws)
        promote = (
            val_points > current_best
            and val_points >= cheap_seed_val_points + min_improvement
        )
        model_id = "model_" + uuid.uuid4().hex[:12]
        model_path = str(Path(args.models) / f"{model_id}.joblib")
        status = "promoted" if promote else "candidate"

        payload = {
            "scaler": components["scaler"],
            "classifier": components["classifier"],
            "seed": seed.to_dict(),
            "seed_id": seed_id,
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(payload, model_path)

        # Two-phase promotion: validate the saved model can be loaded.
        try:
            check = joblib.load(model_path)
            check["scaler"].transform(np.zeros((1, len(check["scaler"].mean_))))
            check["classifier"].predict_proba(np.zeros((1, len(check["scaler"].mean_))))
        except Exception as exc:
            print(f"Model file validation failed for {model_id}: {exc}")
            Path(model_path).unlink(missing_ok=True)
            update_model_status(conn, model_id, "failed")
            mark_trained(conn, seed_id)
            log_training_run(conn, seed_id, model_id, ap, float(metrics["train"]["points"]), val_points, False, train_seconds)
            continue

        add_model(
            conn,
            model_id=model_id,
            seed_id_value=seed_id,
            model_path=model_path,
            train_score=float(metrics["train"]["points"]),
            validation_score=val_points,
            promote=promote,
            status=status,
        )
        mark_trained(conn, seed_id)
        log_training_run(conn, seed_id, model_id, ap, float(metrics["train"]["points"]), val_points, promote, train_seconds)

        if promote:
            current_best = val_points
            print(f"PROMOTED {model_id}: validation_points={val_points:.4f} ap={ap:.4f} hits/draw={metrics['validation']['hits_per_draw']:.3f}")
        else:
            print(f"Kept as non-promoted {model_id}: validation_points={val_points:.4f} "
                  f"(need > best={current_best:.4f} AND >= cheap_seed={cheap_seed_val_points:.4f} + margin={min_improvement:.2f})")


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True)
    parser.add_argument("--state", default="state/ancient_ml.sqlite")
    parser.add_argument("--models", default="models")
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--only-untrained", action="store_true")

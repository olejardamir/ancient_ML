from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import date
import random
import time
from typing import Iterable

from .data import Draw, load_draws, split_time_ordered_3
from .lotto_mapping import cheap_predict
from .registry import add_candidate, best_candidates, connect, log_search_run, row_to_seed
from .scoring import Prediction, score_predictions
from .vedic_engine import SyntheticSeed


def random_seed(rng: random.Random, year_min: int = -5000, year_max: int = 5000) -> SyntheticSeed:
    return SyntheticSeed(
        year=rng.randint(year_min, year_max),
        month=rng.randint(1, 12),
        day=rng.randint(1, 28),
        hour=rng.randint(0, 23),
        minute=rng.randint(0, 59),
        second=rng.randint(0, 59),
        latitude=rng.uniform(-60.0, 60.0),
        longitude=rng.uniform(-180.0, 180.0),
        ayanamsha="LAHIRI",
    )


def mutate_seed(seed: SyntheticSeed, rng: random.Random, scale: float = 1.0) -> SyntheticSeed:
    return SyntheticSeed(
        year=max(-10000, min(10000, seed.year + rng.randint(int(-50 * scale), int(50 * scale)))),
        month=max(1, min(12, seed.month + rng.randint(-1, 1))),
        day=max(1, min(28, seed.day + rng.randint(-2, 2))),
        hour=(seed.hour + rng.randint(-2, 2)) % 24,
        minute=(seed.minute + rng.randint(-10, 10)) % 60,
        second=(seed.second + rng.randint(-10, 10)) % 60,
        latitude=max(-60.0, min(60.0, seed.latitude + rng.uniform(-3.0 * scale, 3.0 * scale))),
        longitude=((seed.longitude + rng.uniform(-6.0 * scale, 6.0 * scale) + 180.0) % 360.0) - 180.0,
        ayanamsha=seed.ayanamsha,
    )


def evaluate_seed(seed: SyntheticSeed, train_draws: list[Draw], validation_draws: list[Draw], validation_weight: float = 2.0) -> dict:
    train_preds = [Prediction.from_lists(*cheap_predict(seed, d.draw_date)) for d in train_draws]
    val_preds = [Prediction.from_lists(*cheap_predict(seed, d.draw_date)) for d in validation_draws]
    train_score = score_predictions(train_preds, train_draws)
    val_score = score_predictions(val_preds, validation_draws)
    total = train_score["points"] + validation_weight * val_score["points"]
    return {
        "train": train_score,
        "validation": val_score,
        "total_score": total,
        "hits_per_draw": val_score["hits_per_draw"],
    }


def _worker_search(
    train_draws: list[Draw],
    validation_draws: list[Draw],
    trials: int,
    top_k: int,
    rng_seed: int,
    parent_seeds: list[SyntheticSeed],
    validation_weight: float,
    scale_schedule: bool,
) -> list[tuple[float, SyntheticSeed, dict]]:
    rng = random.Random(rng_seed)
    best: list[tuple[float, SyntheticSeed, dict]] = [
        (0.0, s, {"from_registry": True}) for s in parent_seeds
    ]

    for i in range(1, trials + 1):
        if best and rng.random() < 0.35:
            parent = rng.choice(best)[1]
            scale = max(0.05, 1.0 - i / max(trials, 1)) if scale_schedule else 1.0
            seed = mutate_seed(parent, rng, scale=scale)
        else:
            seed = random_seed(rng)
        try:
            result = evaluate_seed(seed, train_draws, validation_draws, validation_weight=validation_weight)
        except Exception:
            continue
        score = float(result["total_score"])
        if len(best) < top_k or score > best[-1][0]:
            best.append((score, seed, result))
            best.sort(key=lambda x: x[0], reverse=True)
            best = best[:top_k]

    return best


def search_once(draws: list[Draw], state_path: str, trials: int, top_k: int, rng_seed: int | None = None, validation_weight: float = 2.0, train_fraction: float = 0.70, validation_fraction: float = 0.15, workers: int = 1) -> None:
    conn = connect(state_path)
    train_draws, validation_draws, _test_draws = split_time_ordered_3(draws, train_fraction, validation_fraction)

    # Reload best known candidates from SQLite so mutations continue from
    # previous cycles instead of restarting from scratch each time.
    existing = best_candidates(conn, limit=top_k, only_untrained=False)
    parent_seeds = [row_to_seed(row) for row in existing]

    trials_per_worker = max(1, trials // workers)
    candidates_before = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]

    if workers > 1:
        futures = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for w in range(workers):
                worker_seed = (rng_seed or 0) + w * 1000 + 1
                fut = executor.submit(
                    _worker_search,
                    train_draws,
                    validation_draws,
                    trials_per_worker,
                    top_k,
                    worker_seed,
                    parent_seeds,
                    validation_weight,
                    True,
                )
                futures.append(fut)

        all_results: list[tuple[float, SyntheticSeed, dict]] = []
        for w, fut in enumerate(futures):
            try:
                worker_best = fut.result()
                all_results.extend(worker_best)
                print(f"  Worker {w + 1}/{workers} returned {len(worker_best)} candidates", flush=True)
            except Exception as exc:
                print(f"  Worker {w + 1}/{workers} failed: {exc}", flush=True)

        all_results.sort(key=lambda x: x[0], reverse=True)
        all_results = all_results[:top_k]

        merged = 0
        for score, seed, result in all_results:
            add_candidate(
                conn,
                seed,
                train_score=result["train"]["points"],
                validation_score=result["validation"]["points"],
                total_score=result["total_score"],
                hits_per_draw=result["hits_per_draw"],
            )
            merged += 1
            print(
                f"candidate {seed.iso_datetime()}: val_hits/draw={result['hits_per_draw']:.3f} "
                f"val_points={result['validation']['points']:.2f} lat={seed.latitude:.4f} lon={seed.longitude:.4f}",
                flush=True,
            )
        print(f"Merged {merged} candidates from {workers} workers", flush=True)
    else:
        rng = random.Random(rng_seed)
        best: list[tuple[float, SyntheticSeed, dict]] = [
            (0.0, s, {"from_registry": True}) for s in parent_seeds
        ]

        for i in range(1, trials + 1):
            if best and rng.random() < 0.35:
                parent = rng.choice(best)[1]
                seed = mutate_seed(parent, rng, scale=max(0.05, 1.0 - i / max(trials, 1)))
            else:
                seed = random_seed(rng)
            try:
                result = evaluate_seed(seed, train_draws, validation_draws, validation_weight=validation_weight)
            except Exception:
                continue
            score = float(result["total_score"])
            if len(best) < top_k or score > best[-1][0]:
                best.append((score, seed, result))
                best.sort(key=lambda x: x[0], reverse=True)
                best = best[:top_k]
                sid = add_candidate(
                    conn,
                    seed,
                    train_score=result["train"]["points"],
                    validation_score=result["validation"]["points"],
                    total_score=result["total_score"],
                    hits_per_draw=result["hits_per_draw"],
                )
                print(
                    f"[{i}/{trials}] candidate {sid}: val_hits/draw={result['hits_per_draw']:.3f} "
                    f"val_points={result['validation']['points']:.2f} seed={seed.iso_datetime()} "
                    f"lat={seed.latitude:.4f} lon={seed.longitude:.4f}",
                    flush=True,
                )

    candidates_after = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    log_search_run(conn, trials, candidates_after - candidates_before, 0)


def run_search(args: argparse.Namespace) -> None:
    draws = load_draws(args.data)
    if not draws:
        raise SystemExit("No draws loaded")
    cycle = 0
    while True:
        cycle += 1
        print(f"Search cycle {cycle}: trials={args.trials} workers={args.workers}", flush=True)
        search_once(
            draws, args.state, args.trials, args.top_k, args.seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            workers=args.workers,
        )
        if not args.forever:
            break
        time.sleep(args.sleep)


def add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True)
    parser.add_argument("--state", default="state/ancient_ml.sqlite")
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--forever", action="store_true")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)

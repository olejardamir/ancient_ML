from __future__ import annotations

import argparse
import json

from .data import load_draws, summarize_draws


def run_inspect(args: argparse.Namespace) -> None:
    draws = load_draws(args.data, product=args.product)
    summary = summarize_draws(draws)
    print(json.dumps(summary, indent=2))
    if args.tail and draws:
        for d in draws[-args.tail:]:
            print(f"{d.draw_id} {d.draw_date.isoformat()} {','.join(map(str, d.main_numbers))} bonus={d.bonus}")


def add_inspect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, help="CSV file, for example data/649.csv")
    parser.add_argument("--product", default="649")
    parser.add_argument("--tail", type=int, default=5, help="Print last N rows after the JSON summary")

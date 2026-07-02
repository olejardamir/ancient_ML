from __future__ import annotations

import argparse

from .searcher import add_search_args, run_search
from .trainer import add_train_args, run_train
from .predictor import add_predict_args, run_predict
from .inspect_data import add_inspect_args, run_inspect


def main() -> None:
    parser = argparse.ArgumentParser(prog="ancient_ml", description="Synthetic Vedic Oracle Kundli search/training/prediction framework")
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Run cheap synthetic kundli seed search")
    add_search_args(p_search)
    p_search.set_defaults(func=run_search)

    p_train = sub.add_parser("train", help="Train/promote ML rankers from candidate seeds")
    add_train_args(p_train)
    p_train.set_defaults(func=run_train)

    p_predict = sub.add_parser("predict", help="Predict Lotto 6/49 numbers for a draw date")
    add_predict_args(p_predict)
    p_predict.set_defaults(func=run_predict)

    p_inspect = sub.add_parser("inspect-data", help="Check that a Lotto 6/49 CSV loads correctly")
    add_inspect_args(p_inspect)
    p_inspect.set_defaults(func=run_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Compare named full-model variants on a small set of units."""

import argparse

import polars.selectors as cs

from gain_glm import CVConfig, Dropout, FitConfig
from gain_glm.dynamic_routing import MODELS, compare_models, parse_dropout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nwb-path", required=True)
    parser.add_argument(
        "--model", action="append", choices=MODELS, dest="models", required=True
    )
    parser.add_argument("--unit", action="append", dest="units")
    parser.add_argument("--unit-limit", type=int, default=8)
    parser.add_argument("--dropout", action="append", type=parse_dropout)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int)
    parser.add_argument("--max-iter", type=int, default=50)
    args = parser.parse_args()

    table = compare_models(
        args.nwb_path,
        [MODELS[name] for name in args.models],
        unit_ids=args.units,
        unit_limit=args.unit_limit,
        dropouts=args.dropout or (Dropout.gain("context"),),
        fit=FitConfig(max_iter=args.max_iter),
        cv=CVConfig(folds=args.folds, seed=args.fold_seed),
    )
    print(table)
    print(
        table.group_by("model")
        .agg(cs.numeric().mean())
        .sort("cv_r2", descending=True)
    )


if __name__ == "__main__":
    main()

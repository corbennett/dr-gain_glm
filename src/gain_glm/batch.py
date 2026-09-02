"""Batch fitting, model comparison, and CLI orchestration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from .design import PreparedDesign, compile_design
from .dynamic_routing import (
    DEFAULT_MODEL,
    MODELS,
    QC_COLUMN,
    load_session,
    load_unit_target,
    model_data,
    prepare,
    qc_unit_ids,
)
from .evaluation import CVConfig
from .model import Dropout, FitConfig, ModelSpec


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _evaluate_unit(
    unit_id: str,
    y: np.ndarray,
    prepared: PreparedDesign,
    fit: FitConfig,
    cv: CVConfig,
    dropouts: Sequence[Dropout],
) -> tuple[str, dict]:
    with threadpool_limits(1):
        result = prepared.evaluate(y, fit=fit, cv=cv, dropouts=dropouts)
    return unit_id, result.to_dict()


def fit_session(
    nwb_path: str,
    session_id: str,
    output_dir: str | Path,
    *,
    model: ModelSpec = DEFAULT_MODEL,
    dropouts: Sequence[Dropout] | None = None,
    fit: FitConfig | None = None,
    cv: CVConfig | None = None,
    n_jobs: int = -1,
    limit: int | None = None,
    qc_column: str = QC_COLUMN,
    overwrite: bool = False,
) -> Path:
    """Fit every selected unit while reusing one prepared session design."""
    output_path = Path(output_dir) / f"{session_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        print(f"[{session_id}] {output_path} exists; skipping")
        return output_path

    units = qc_unit_ids(nwb_path, qc_column=qc_column)
    if limit is not None:
        units = units[:limit]
    session = load_session(nwb_path, dt=model.dt)
    prepared = prepare(session, model)
    targets = {unit: load_unit_target(session, unit) for unit in units}
    fit_config = FitConfig() if fit is None else fit
    cv_config = CVConfig() if cv is None else cv
    dropout_requests = model.dropouts if dropouts is None else tuple(dropouts)
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_evaluate_unit)(
            unit, targets[unit], prepared, fit_config, cv_config, dropout_requests
        )
        for unit in units
    )
    payload = {
        "session_id": session_id,
        "model": model.name,
        "dt": model.dt,
        "fit_window": model.fit_window,
        "fit_events": model.fit_events,
        "units": dict(results),
    }
    with output_path.open("w") as stream:
        json.dump(payload, stream, default=_json_default)
    return output_path


def compare_models(
    nwb_path: str,
    models: Sequence[ModelSpec],
    *,
    unit_ids: Sequence[str] | None = None,
    dropouts: Sequence[Dropout] | None = None,
    fit: FitConfig | None = None,
    cv: CVConfig | None = None,
    unit_limit: int = 8,
) -> pl.DataFrame:
    """Compare full-model variants using one session load and target cache."""
    if not models:
        raise ValueError("at least one model is required")
    reference = models[0]
    if any(not np.isclose(model.dt, reference.dt) for model in models[1:]):
        raise ValueError("compared models must use the same dt")
    if any(
        (model.fit_window, model.fit_events)
        != (reference.fit_window, reference.fit_events)
        for model in models[1:]
    ):
        raise ValueError("compared models must use the same fit window and events")
    session = load_session(nwb_path, dt=reference.dt)
    inputs = model_data(session)
    prepared = {model.name: compile_design(model, inputs) for model in models}
    if len(prepared) != len(models):
        raise ValueError("model names must be unique")
    units = (
        list(unit_ids) if unit_ids is not None else qc_unit_ids(nwb_path)[:unit_limit]
    )
    targets = {unit: load_unit_target(session, unit) for unit in units}
    fit_config = FitConfig() if fit is None else fit
    cv_config = CVConfig() if cv is None else cv

    rows: list[dict[str, Any]] = []
    for model in models:
        dropout_requests = model.dropouts if dropouts is None else tuple(dropouts)
        for unit in units:
            with threadpool_limits(1):
                result = prepared[model.name].evaluate(
                    targets[unit],
                    fit=fit_config,
                    cv=cv_config,
                    dropouts=dropout_requests,
                )
            row: dict[str, Any] = {
                "model": model.name,
                "unit_id": unit,
                "train_r2": result.train_r2,
                "cv_r2": result.cv.r2,
                "worst_fold_r2": float(np.min(result.cv.r2_per_fold)),
                "n_iter": result.fit.n_iter,
                "converged": result.fit.converged,
                "cv_converged": result.cv.converged,
                "n_converged_folds": int(np.sum(result.cv.converged_per_fold)),
            }
            for name, dropout in result.dropouts.items():
                row[f"delta_r2_{name}"] = dropout.delta_r2
            rows.append(row)
    return pl.DataFrame(rows)


def parse_dropout(value: str) -> Dropout:
    """Parse ``gain:name``, ``group:name``, or ``predictors:a,b``."""
    try:
        kind, names = value.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "dropouts use gain:name, group:name, or predictors:a,b"
        ) from error
    if kind == "gain":
        return Dropout.gain(names)
    if kind == "group":
        return Dropout.group(names)
    if kind == "predictors":
        predictors = tuple(name for name in names.split(",") if name)
        return Dropout.predictors(*predictors)
    raise argparse.ArgumentTypeError(f"unknown dropout kind {kind!r}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nwb-path", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=MODELS, default="default")
    dropout_options = parser.add_mutually_exclusive_group()
    dropout_options.add_argument(
        "--dropout",
        action="append",
        type=parse_dropout,
        help="Repeatable: gain:name, group:name, or predictors:a,b",
    )
    dropout_options.add_argument(
        "--no-dropouts",
        action="store_true",
        help="Disable the model's declared dropout comparisons",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int)
    parser.add_argument("--max-iter", type=int, default=FitConfig().max_iter)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--qc-column", default=QC_COLUMN)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    fit_session(
        args.nwb_path,
        args.session_id,
        args.output_dir,
        model=MODELS[args.model],
        dropouts=() if args.no_dropouts else args.dropout,
        fit=FitConfig(max_iter=args.max_iter),
        cv=CVConfig(folds=args.folds, seed=args.fold_seed),
        n_jobs=args.n_jobs,
        limit=args.limit,
        qc_column=args.qc_column,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

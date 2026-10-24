"""Run preference versus baseline-context comparisons on the cached dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

from . import ANALYSIS_VERSION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/modality_context/full_expanded"),
    )
    p.add_argument(
        "--selection-dir",
        type=Path,
        default=Path("results/responsive_preferences/full"),
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("results/preference_context/full")
    )
    p.add_argument("--permutations", type=int, default=1999)
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--session", action="append", help="Subset for a smoke test only")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    if args.permutations < 1 or args.threads < 1:
        raise ValueError("Permutation and thread counts must be positive")
    source, selection_dir, out = [
        x.resolve() for x in (args.source_dir, args.selection_dir, args.output_dir)
    ]
    if out in (source, selection_dir):
        raise ValueError("Output must be separate from input datasets")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    os.environ.setdefault("POLARS_MAX_THREADS", str(args.threads))
    import numpy as np
    import polars as pl
    from threadpoolctl import threadpool_limits

    from gain_glm.context_preferences.data import atomic_json, atomic_parquet

    from .analysis import aggregate, evaluate_session

    for directory in (source, selection_dir):
        if json.loads((directory / "status.json").read_text())["state"] != "complete":
            raise ValueError(f"Input run is incomplete: {directory}")
    selection_config = json.loads((selection_dir / "configuration.json").read_text())
    if Path(selection_config["source_dir"]).resolve() != source:
        raise ValueError("Selected cells came from a different source dataset")
    units = pl.read_parquet(source / "selected_units.parquet").sort(
        "session_id", "unit_id"
    )
    selection = pl.read_parquet(selection_dir / "selected_neurons.parquet")
    all_sessions = sorted(units["session_id"].unique())
    sessions = sorted(args.session or all_sessions)
    if set(sessions) - set(all_sessions):
        raise ValueError("Unknown session requested")
    config = {
        "analysis_version": ANALYSIS_VERSION,
        "source_dir": str(source),
        "selection_dir": str(selection_dir),
        "permutations": args.permutations,
        "seed": args.seed,
        "session_subset": args.session,
        "minimum_per_group": 3,
        "matching_caliper": 1.5,
        "alpha": 0.05,
        "primary_selection_half": 0,
        "primary_preference_quarter": 0,
        "baseline_test": "OLS_cubic_drift_CR2_context_block_Satterthwaite",
        "CR2_working_within_block_correlation": 0.5,
    }
    path = out / "configuration.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Configuration changed; use a fresh output directory")
    atomic_json(path, config)
    code = {f.name: f.read_bytes() for f in Path(__file__).parent.glob("*.py")}
    for module, names in (
        ("context_preferences", ("statistics.py", "data.py")),
        ("responsive_preferences", ("statistics.py", "analysis.py")),
    ):
        for name in names:
            code[f"{module}_{name}"] = (
                Path(__file__).parents[1] / module / name
            ).read_bytes()
    files = [
        source / "selected_units.parquet",
        source / "configuration.json",
        selection_dir / "selected_neurons.parquet",
        selection_dir / "configuration.json",
        *[source / "counts" / f"{sid}.npz" for sid in sessions],
    ]
    provenance = {
        "source_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in code.items()
        },
        "input_sha256": {
            str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files
        },
        "python": sys.version,
        "command": sys.argv,
        "packages": {
            k: importlib.metadata.version(k)
            for k in ("numpy", "scipy", "polars", "matplotlib")
        },
    }
    path = out / "provenance.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if any(previous[k] != provenance[k] for k in ("source_sha256", "input_sha256")):
            raise ValueError("Code or inputs changed; use a fresh output directory")
    else:
        atomic_json(path, provenance)
        (out / "source_snapshot").mkdir(exist_ok=True)
        for name, content in code.items():
            (out / "source_snapshot" / name).write_bytes(content)
    started, done = time.time(), []

    def status(state, **extra):
        atomic_json(
            out / "status.json",
            {
                "state": state,
                "pid": os.getpid(),
                "completed_sessions": done,
                "requested_sessions": len(sessions),
                "elapsed_seconds": time.time() - started,
                **extra,
            },
        )

    (out / "sessions").mkdir(exist_ok=True)
    frames = []
    try:
        with threadpool_limits(limits=args.threads):
            for i, sid in enumerate(sessions, 1):
                tick = time.time()
                cache = out / "sessions" / f"{sid}.parquet"
                empty = cache.with_suffix(".empty.json")
                if not cache.exists() and not empty.exists():
                    with np.load(
                        source / "counts" / f"{sid}.npz", allow_pickle=False
                    ) as data:
                        arrays = {k: data[k] for k in data.files}
                    result = evaluate_session(
                        selection.filter(pl.col("session_id") == sid),
                        units.filter(pl.col("session_id") == sid),
                        arrays,
                        args.seed,
                        args.permutations,
                    )
                    if result is None:
                        atomic_json(empty, {"no_responsive_cells": True})
                    else:
                        atomic_parquet(cache, result)
                if cache.exists():
                    frames.append(pl.read_parquet(cache))
                done.append(sid)
                print(
                    f"[{i}/{len(sessions)}] {sid}: complete ({time.time() - tick:.1f}s)",
                    flush=True,
                )
                status("evaluating")
            if not frames:
                raise ValueError("No responsive cells in requested dataset")
            status("aggregating")
            result = aggregate(pl.concat(frames), out, args.seed)
            if not args.no_plots:
                status("rendering")
                from .plots import report

                report(out, *result)
            status("complete", failures=[])
            print(
                f"Finished {len(done)}/{len(sessions)} sessions in {time.time() - started:.1f}s: {out}",
                flush=True,
            )
    except Exception as error:
        status("failed", failures=[str(error)])
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

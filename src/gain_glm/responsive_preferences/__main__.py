"""Run responsive-cell preference comparisons from the cached 17-area dataset."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/modality_context/full_expanded"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/responsive_preferences/full")
    )
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--response-alpha", type=float, default=0.05)
    parser.add_argument(
        "--session",
        action="append",
        help="Evaluation subset for a smoke run; screening FDR still uses all sessions",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.permutations < 1 or args.threads < 1 or not 0 < args.response_alpha < 1:
        raise ValueError("Invalid permutation count, thread count, or alpha")
    source, out = args.source_dir.resolve(), args.output_dir.resolve()
    if source == out:
        raise ValueError("Use a separate output directory")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    os.environ.setdefault("POLARS_MAX_THREADS", str(args.threads))
    import numpy as np
    import polars as pl
    from threadpoolctl import threadpool_limits

    from gain_glm.context_preferences.data import atomic_json, atomic_parquet

    from .analysis import (
        aggregate,
        evaluate_session,
        prepare_session,
        select_responsive,
    )

    source_status = json.loads((source / "status.json").read_text())
    if source_status["state"] != "complete":
        raise ValueError("The source dataset must be complete")
    units = pl.read_parquet(source / "selected_units.parquet").sort(
        "session_id", "unit_id"
    )
    all_sessions = sorted(source_status["completed_sessions"])
    requested = sorted(args.session or all_sessions)
    if set(requested) - set(all_sessions):
        raise ValueError("Requested session absent from source dataset")
    config = {
        "analysis_version": ANALYSIS_VERSION,
        "source_dir": str(source),
        "permutations": args.permutations,
        "seed": args.seed,
        "response_alpha": args.response_alpha,
        "session_subset": args.session,
        "primary_selection_half": 0,
        "normalization_floor_hz": 1,
        "responsiveness_test": "paired_rate_t_two_sided_bonferroni_conditions_then_BH_area_cohort_window_split",
    }
    config_path = out / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Configuration changed; choose a new output directory")
    atomic_json(config_path, config)
    code = {p.name: p.read_bytes() for p in Path(__file__).parent.glob("*.py")}
    helper = Path(__file__).parents[1] / "context_preferences" / "statistics.py"
    code["context_preferences_statistics.py"] = helper.read_bytes()
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in code.items()}
    input_hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in [
            "selected_units.parquet",
            "configuration.json",
            *[f"counts/{sid}.npz" for sid in all_sessions],
        ]
    }
    provenance = {
        "source_sha256": hashes,
        "input_sha256": input_hashes,
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
            raise ValueError(
                "Analysis code or input changed; choose a fresh output directory"
            )
    else:
        atomic_json(path, provenance)
        (out / "source_snapshot").mkdir(exist_ok=True)
        for name, content in code.items():
            (out / "source_snapshot" / name).write_bytes(content)
    started = time.time()
    done, failures = [], []

    def status(state, **extra):
        atomic_json(
            out / "status.json",
            {
                "state": state,
                "pid": os.getpid(),
                "completed_sessions": done,
                "requested_sessions": len(requested),
                "failures": failures,
                "elapsed_seconds": time.time() - started,
                **extra,
            },
        )

    def load_arrays(sid):
        with np.load(source / "counts" / f"{sid}.npz", allow_pickle=False) as cached:
            return {k: cached[k] for k in cached.files}

    with threadpool_limits(limits=args.threads):
        try:
            screen_dir, eval_dir = out / "screening", out / "sessions"
            screen_dir.mkdir(exist_ok=True)
            eval_dir.mkdir(exist_ok=True)
            for number, sid in enumerate(all_sessions, 1):
                path = screen_dir / f"{sid}.parquet"
                if not path.exists():
                    frame = prepare_session(
                        units.filter(pl.col("session_id") == sid),
                        load_arrays(sid),
                        args.seed,
                    )
                    atomic_parquet(path, frame)
                print(f"Screening [{number}/{len(all_sessions)}] {sid}", flush=True)
                status("screening", screened_sessions=number)
            selection = pl.concat(
                [pl.read_parquet(screen_dir / f"{sid}.parquet") for sid in all_sessions]
            )
            selection = select_responsive(selection, args.response_alpha)
            atomic_parquet(out / "selection_all_units.parquet", selection)
            print(
                selection.group_by("window", "selection_half", "cohort").agg(
                    pl.len().alias("tested"),
                    pl.col("responsive").sum().alias("responsive"),
                ),
                flush=True,
            )
            unit_frames, stat_frames = [], []
            for number, sid in enumerate(requested, 1):
                tick = time.time()
                up, sp, empty = (
                    eval_dir / f"{sid}_units.parquet",
                    eval_dir / f"{sid}_statistics.parquet",
                    eval_dir / f"{sid}_empty.json",
                )
                if not empty.exists() and not (up.exists() and sp.exists()):
                    uf, sf = evaluate_session(
                        selection.filter(pl.col("session_id") == sid),
                        units.filter(pl.col("session_id") == sid),
                        load_arrays(sid),
                        args.seed,
                        args.permutations,
                    )
                    if uf is None:
                        atomic_json(
                            empty, {"session_id": sid, "no_selected_cells": True}
                        )
                    else:
                        atomic_parquet(up, uf)
                        atomic_parquet(sp, sf)
                if not empty.exists():
                    unit_frames.append(pl.read_parquet(up))
                    stat_frames.append(pl.read_parquet(sp))
                done.append(sid)
                print(
                    f"Evaluation [{number}/{len(requested)}] {sid}: complete ({time.time() - tick:.1f}s)",
                    flush=True,
                )
                status("evaluating")
            if not unit_frames:
                raise ValueError("No responsive cells in requested sessions")
            selected_scope = selection.filter(pl.col("session_id").is_in(requested))
            result = aggregate(
                selected_scope,
                pl.concat(unit_frames),
                pl.concat(stat_frames),
                out,
                args.seed,
            )
            if not args.no_plots:
                status("rendering")
                from .plots import report

                report(out, *result)
            status("complete")
            print(
                f"Finished {len(done)}/{len(requested)} sessions in {time.time() - started:.1f}s: {out}",
                flush=True,
            )
        except Exception as error:
            failures.append({"error": str(error), "traceback": traceback.format_exc()})
            status("failed")
            raise


if __name__ == "__main__":
    main()

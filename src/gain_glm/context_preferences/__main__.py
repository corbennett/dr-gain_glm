"""Run with python -m gain_glm.context_preferences (see analysis README)."""

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

from . import ANALYSIS_VERSION, AREA_GROUPS, INCLUDED_AREAS


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/modality_context/full_expanded"),
    )
    p.add_argument(
        "--metadata-dir", type=Path, default=Path("results/modality_context/discovery")
    )
    p.add_argument("--version", default="v0.0.289")
    p.add_argument("--minimum-area-neurons", type=int, default=1000)
    p.add_argument(
        "--include-areas",
        nargs="*",
        default=list(INCLUDED_AREAS),
        help="Areas to include below the neuron threshold (default: MRN SCm ACAd ACAv ORBvl)",
    )
    p.add_argument("--minimum-trials-per-half", type=int, default=5)
    p.add_argument("--permutations", type=int, default=1999)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument(
        "--anatomy", type=Path, help="Optional unit_id-keyed CSV or parquet annotations"
    )
    p.add_argument(
        "--session",
        action="append",
        help="Restrict to named session(s), for validation",
    )
    p.add_argument(
        "--limit-sessions",
        type=int,
        help="Validation only; the area selection still uses all sessions",
    )
    p.add_argument("--discover-only", action="store_true")
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild tables/figures from completed sessions",
    )
    p.add_argument("--no-plots", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if (
        min(
            args.permutations,
            args.minimum_area_neurons,
            args.minimum_trials_per_half,
            args.threads,
        )
        < 1
    ):
        raise ValueError("Counts and thread limits must be positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    os.environ.setdefault("LAZYNWB_CATALOG_CACHE_PATH", str(out / "nwb_catalog.sqlite"))
    os.environ.setdefault("POLARS_MAX_THREADS", str(args.threads))
    import polars as pl
    from threadpoolctl import threadpool_limits

    from .analysis import aggregate, analyze_session
    from .data import (
        atomic_json,
        atomic_parquet,
        build_manifest,
        count_session,
        fetch_metadata,
    )

    config = {
        "analysis_version": ANALYSIS_VERSION,
        "datacube_version": args.version,
        "minimum_area_neurons": args.minimum_area_neurons,
        "included_areas": sorted(set(args.include_areas)),
        "area_groups": {key: list(value) for key, value in AREA_GROUPS.items()},
        "minimum_trials_per_half": args.minimum_trials_per_half,
        "permutations": args.permutations,
        "seed": args.seed,
        "anatomy": str(args.anatomy.resolve()) if args.anatomy else None,
        "session_subset": args.session,
        "limit_sessions": args.limit_sessions,
    }
    if args.anatomy:
        config["anatomy_sha256"] = hashlib.sha256(args.anatomy.read_bytes()).hexdigest()
    config_path = out / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(
            "Run configuration changed. Use a new output directory to avoid mixing analyses."
        )
    atomic_json(config_path, config)
    source_files = {
        name: (Path(__file__).parent / name).read_text()
        for name in ("data.py", "statistics.py", "analysis.py")
    }
    hashes = {
        name: hashlib.sha256(text.encode()).hexdigest()
        for name, text in source_files.items()
    }
    provenance_path = out / "provenance.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text())["source_sha256"] != hashes:
            raise ValueError("Analysis source changed; use a fresh run directory")
    else:
        atomic_json(
            provenance_path,
            {
                "source_sha256": hashes,
                "python": sys.version,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "numpy",
                        "scipy",
                        "polars",
                        "dr-datacube",
                        "lazynwb",
                        "matplotlib",
                    )
                },
                "command": sys.argv,
            },
        )
        snapshot = out / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for source in Path(__file__).parent.glob("*.py"):
            (snapshot / source.name).write_text(source.read_text())
    started = time.time()
    with threadpool_limits(limits=args.threads):
        fetch_metadata(args.metadata_dir, args.version)
        manifest, units, trials = build_manifest(
            args.metadata_dir,
            out,
            args.minimum_area_neurons,
            args.anatomy,
            args.include_areas,
        )
        included = manifest.filter(pl.col("selection_status") == "included")
        if args.session:
            missing = set(args.session) - set(included["session_id"])
            if missing:
                raise ValueError(f"Requested sessions not eligible: {sorted(missing)}")
            included = included.filter(pl.col("session_id").is_in(args.session))
        if args.limit_sessions:
            included = included.head(args.limit_sessions)
        print(
            f"Selected {included.height} sessions; {units['structure'].n_unique()} areas; "
            f"{units.height:,} QC-passing units across the full manifest",
            flush=True,
        )
        print(
            manifest.group_by("cohort", "selection_status")
            .len()
            .sort("cohort", "selection_status"),
            flush=True,
        )
        if args.discover_only:
            return
        done, failures = [], []
        session_dir = out / "sessions"
        session_dir.mkdir(exist_ok=True)
        for number, row in enumerate(included.iter_rows(named=True), 1):
            sid = row["session_id"]
            unit_path, stat_path = (
                session_dir / f"{sid}_units.parquet",
                session_dir / f"{sid}_statistics.parquet",
            )
            tick = time.time()
            try:
                if not (unit_path.exists() and stat_path.exists()):
                    if args.summarize_only:
                        raise FileNotFoundError("Session has not completed")
                    session_units = units.filter(pl.col("session_id") == sid)
                    session_trials = trials.filter(pl.col("session_id") == sid)
                    print(
                        f"[{number}/{included.height}] {sid}: loading {session_units.height} neurons",
                        flush=True,
                    )
                    arrays = count_session(
                        row, session_units, session_trials, out / "counts", args.seed
                    )
                    print(
                        f"[{number}/{included.height}] {sid}: analyzing {len(arrays['times'])} trials",
                        flush=True,
                    )
                    unit_frame, stat_frame = analyze_session(
                        row,
                        session_units,
                        arrays,
                        args.permutations,
                        args.seed,
                        args.minimum_trials_per_half,
                    )
                    atomic_parquet(unit_path, unit_frame)
                    atomic_parquet(stat_path, stat_frame)
                    del arrays, unit_frame, stat_frame
                done.append(sid)
                print(
                    f"[{number}/{included.height}] {sid}: complete ({time.time() - tick:.1f}s)",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 -- record isolated session failures and continue the batch
                failures.append(
                    {
                        "session_id": sid,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(
                    f"[{number}/{included.height}] {sid}: FAILED: {error}", flush=True
                )
                traceback.print_exc()
            atomic_json(
                out / "status.json",
                {
                    "state": "running",
                    "pid": os.getpid(),
                    "completed_sessions": done,
                    "failures": failures,
                    "requested_sessions": included.height,
                    "elapsed_seconds": time.time() - started,
                },
            )
        if not done:
            raise RuntimeError("No sessions completed; see status.json")
        print("Aggregating completed sessions and writing reports", flush=True)
        unit_stats = pl.concat(
            [pl.read_parquet(session_dir / f"{s}_units.parquet") for s in done],
            how="diagonal_relaxed",
        )
        session_stats = pl.concat(
            [pl.read_parquet(session_dir / f"{s}_statistics.parquet") for s in done],
            how="diagonal_relaxed",
        )
        area, sessions, unit_stats = aggregate(
            session_stats, unit_stats, out, args.seed
        )
        if not args.no_plots:
            from .plots import report

            report(out, area, sessions, unit_stats, manifest, failures)
        atomic_json(
            out / "status.json",
            {
                "state": "complete" if not failures else "incomplete",
                "pid": os.getpid(),
                "completed_sessions": done,
                "failures": failures,
                "requested_sessions": included.height,
                "elapsed_seconds": time.time() - started,
            },
        )
        print(
            f"Finished: {len(done)}/{included.height} sessions, {time.time() - started:.1f}s; {out}",
            flush=True,
        )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()

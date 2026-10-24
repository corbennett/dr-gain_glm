"""Compare held-out stimulus preferences with the notebook's saved GLM baseline labels."""

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
        "--glm-parquet",
        type=Path,
        default=Path("/Users/corbettb/Documents/scratch/gain_glm_df_100626.parquet"),
    )
    parser.add_argument(
        "--preference-dir",
        type=Path,
        default=Path("results/responsive_preferences/full"),
    )
    parser.add_argument(
        "--baseline-dir", type=Path, default=Path("results/preference_context/full")
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("results/modality_context/full_expanded"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/preference_context/glm")
    )
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    inputs = [
        args.glm_parquet.resolve(),
        args.preference_dir.resolve() / "selected_neurons.parquet",
        args.baseline_dir.resolve() / "selected_neurons.parquet",
        args.metadata_dir.resolve() / "selected_units.parquet",
    ]
    if out in [f.parent for f in inputs]:
        raise ValueError("Output directory must be separate from input datasets")
    for directory in (args.preference_dir, args.baseline_dir, args.metadata_dir):
        if json.loads((directory / "status.json").read_text())["state"] != "complete":
            raise ValueError(f"Incomplete input: {directory}")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("POLARS_MAX_THREADS", str(args.threads))
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import polars as pl
    from threadpoolctl import threadpool_limits

    from gain_glm.context_preferences.data import atomic_json

    from .analysis import GLM_COLUMNS, compare, link_data

    config = {
        "analysis_version": ANALYSIS_VERSION,
        "glm_parquet": str(inputs[0]),
        "preference_dir": str(args.preference_dir.resolve()),
        "baseline_dir": str(args.baseline_dir.resolve()),
        "metadata_dir": str(args.metadata_dir.resolve()),
        "seed": args.seed,
        "baseline_primary": "saved_GLM_permutation_p_less_than_0.05_notebook_definition",
        "baseline_FDR": "BH_all_available_QC_cells_per_area_cohort",
        "minimum_group_cells": 3,
        "matching_caliper": 1.5,
        "preference_primary_selection_half": 0,
        "independent_baseline_trials": False,
    }
    code = {p.name: p.read_bytes() for p in Path(__file__).parent.glob("*.py")}
    for rel in (
        "preference_context/statistics.py",
        "preference_context/plots.py",
        "responsive_preferences/statistics.py",
        "context_preferences/statistics.py",
    ):
        code[rel.replace("/", "_")] = (Path(__file__).parents[2] / rel).read_bytes()

    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                h.update(chunk)
        return h.hexdigest()

    provenance = {
        "source_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in code.items()},
        "input_sha256": {str(p): digest(p) for p in inputs},
        "python": sys.version,
        "command": sys.argv,
        "packages": {
            k: importlib.metadata.version(k)
            for k in ("numpy", "scipy", "polars", "matplotlib")
        },
    }
    for name, content in (
        ("configuration.json", config),
        ("provenance.json", provenance),
    ):
        path = out / name
        if path.exists():
            previous = json.loads(path.read_text())
            keys = (
                content
                if name == "configuration.json"
                else ("source_sha256", "input_sha256")
            )
            if any(previous[k] != content[k] for k in keys):
                raise ValueError(
                    "Configuration, code, or data changed; use a fresh output directory"
                )
        else:
            atomic_json(path, content)
    (out / "source_snapshot").mkdir(exist_ok=True)
    for name, content in code.items():
        (out / "source_snapshot" / name).write_bytes(content)
    started = time.time()

    def status(state, **kw):
        atomic_json(
            out / "status.json",
            {
                "state": state,
                "pid": os.getpid(),
                "elapsed_seconds": time.time() - started,
                **kw,
            },
        )

    try:
        with threadpool_limits(limits=args.threads):
            status("linking")
            glm = pl.read_parquet(
                inputs[0], columns=["session_id", "unit_id", *GLM_COLUMNS]
            )
            units, coverage, universe = link_data(
                pl.read_parquet(inputs[3]),
                pl.read_parquet(inputs[1]),
                pl.read_parquet(inputs[2]),
                glm,
            )
            coverage.write_csv(out / "glm_coverage.csv")
            universe.write_parquet(out / "glm_qc_cells.parquet")
            print(
                coverage.group_by("cohort").agg(
                    pl.col("n_qc").sum(),
                    pl.col("n_glm").sum(),
                    pl.col("n_converged").sum(),
                ),
                flush=True,
            )
            status("comparing")
            summary = compare(units, out, args.seed)
            status("rendering")
            from .plots import report

            report(out, units, summary, coverage)
            status(
                "complete",
                failures=[],
                n_linked_unique_cells=units["unit_id"].n_unique(),
                n_expert_sessions=universe["session_id"].n_unique(),
            )
            print(
                f"Finished GLM association analysis in {time.time() - started:.1f}s: {out}",
                flush=True,
            )
    except Exception as error:
        status("failed", failures=[str(error)])
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

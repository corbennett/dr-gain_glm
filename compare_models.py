"""Compare model iterations on a handful of cells from one session.

A lightweight bench for iterating on the bilinear gain GLM. It loads each
cell's spike train once, builds each distinct design once, then fits every
(variant x cell) combination and returns a tidy table of metrics — so you can
eyeball how a change to the model, the peri-event window, or the CV scheme
moves train_r2 / cv_r2 / delta_r2 before launching a full SLURM sweep.

Usage:
    python compare_models.py                 # run the default VARIANTS, print + save
    # or, in a notebook:
    from compare_models import compare, VARIANTS
    table = compare(VARIANTS)
    table.pivot(values="cv_r2", index="unit_id", on="variant")

A "variant" is a dict:
    {
      "name":      str,                       # label in the output table
      "design_fn": callable(SessionData) -> design dict,  # optional; defaults
                                              # to run_fit.assemble_design.
                                              # Swap in your own builder to test
                                              # a different session design (new
                                              # predictors, windows, alphas, ...).
                                              # Variants sharing a design_fn share
                                              # one design build.
      "fit_kwargs": dict,                     # optional; passed to model.fit_summary.
                                              # e.g. {"fold_seed": 0}, {"fit_mask": None}.
    }
The expensive NWB load happens once per session (load_session_data); every
design_fn is then built from that shared SessionData, so comparing many designs
never re-reads from S3. By default each variant fits with remove_gains=["context"]
and the design's own fit_mask (peri-stimulus window). Override fit_mask=None in
fit_kwargs to fit all bins, or fold_seed=<int> for random (not contiguous) folds.

To test a new design, copy run_fit.assemble_design, tweak it, and pass it as a
variant's design_fn -- see `example_design_no_lp` below.
"""

import time

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from run_fit import (
    DT,
    assemble_design,
    get_qc_pass_unit_ids,
    load_session_data,
    load_y,
)
from designs import DESIGNS

# ---------------------------------------------------------------------------
# What to fit
# ---------------------------------------------------------------------------
# A handful of cells from one session for quick iteration. Paste your own list
# here; if left as None the harness grabs the first N qc-pass units.
SESSION_ID = "668755_2023-08-31"
NWB_PATH = (
    "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.272/"
    f"{SESSION_ID}.nwb"
)
UNIT_IDS: list[str] | None = [
    "668755_2023-08-31_B-85",
    "668755_2023-08-31_B-89",
    "668755_2023-08-31_B-96",
    "668755_2023-08-31_B-97",
    "668755_2023-08-31_B-229",
    "668755_2023-08-31_B-283",
    "668755_2023-08-31_B-429",
    "668755_2023-08-31_C-301",
    "668755_2023-08-31_C-305",
    "668755_2023-08-31_C-516",
    "668755_2023-08-31_C-623",
]

N_UNITS_IF_NONE = 8

# ---------------------------------------------------------------------------
# Model iterations to compare. Edit freely.
# ---------------------------------------------------------------------------
# Designs live in designs.py (the DESIGNS registry); reference them by name.
# To add a new one, add a kernel-spec entry there — no builder copying.
# Variants sharing a design_fn share one design build; fit_kwargs pass through
# to model.fit_summary (e.g. fold_seed, fit_mask).
VARIANTS: list[dict] = [
    {"name": "default", "design_fn": DESIGNS["default"], "fit_kwargs": {"fold_seed": 0}},
    {"name": "no_hit_long_stim", "design_fn": DESIGNS["no_hit_long_stim"], "fit_kwargs": {"fold_seed": 0}},
    {"name": "all_response", "design_fn": DESIGNS["all_response"], "fit_kwargs": {"fold_seed": 0}},
]


def _fit_one(model, y, sd, fit_kwargs):
    with threadpool_limits(1):
        return model.fit_summary(
            y=y, trial_idx=sd["trial_idx"], design=sd["design"], **fit_kwargs
        )


def compare(variants=VARIANTS, *, nwb_path=NWB_PATH, unit_ids=UNIT_IDS,
            dt=DT) -> pl.DataFrame:
    """Fit every (variant x cell) and return one row per fit.

    The session's NWB data is loaded once (load_session_data); each distinct
    `design_fn` is built once from that shared data; spike trains are loaded
    once per cell. So adding variants — even structurally different designs —
    is nearly free and never re-reads from S3.
    """
    if unit_ids is None:
        unit_ids = get_qc_pass_unit_ids(nwb_path)[:N_UNITS_IF_NONE]
    print(f"comparing {len(variants)} variants x {len(unit_ids)} cells "
          f"from {nwb_path}")

    # Load the session's NWB streams once — shared by every design_fn.
    t0 = time.perf_counter()
    data = load_session_data(nwb_path, dt=dt)
    print(f"  loaded session data in {time.perf_counter() - t0:.1f}s")

    # Build each distinct design once (keyed by the design_fn object).
    design_cache: dict = {}

    def get_sd(design_fn):
        if design_fn not in design_cache:
            t0 = time.perf_counter()
            design_cache[design_fn] = design_fn(data)
            print(f"  built design via {design_fn.__name__} "
                  f"in {time.perf_counter() - t0:.1f}s")
        return design_cache[design_fn]

    # Load every cell's spike train once (independent of the model variant).
    t0 = time.perf_counter()
    y_by_unit = {
        uid: load_y(nwb_path, uid, data.task_start_time,
                    data.task_end_time, dt)
        for uid in unit_ids
    }
    print(f"  loaded {len(unit_ids)} spike trains in "
          f"{time.perf_counter() - t0:.1f}s")

    rows = []
    for v in variants:
        sd = get_sd(v.get("design_fn", assemble_design))
        fit_kwargs = {"remove_gains": ["context"], **v.get("fit_kwargs", {})}
        # Default to the design's peri-stimulus fit_mask unless the variant
        # explicitly set fit_mask (to None or otherwise).
        if "fit_mask" not in fit_kwargs:
            fit_kwargs["fit_mask"] = sd.get("fit_mask")
        for uid in unit_ids:
            t0 = time.perf_counter()
            s = _fit_one(sd["model"], y_by_unit[uid], sd, fit_kwargs)
            rows.append({
                "variant": v["name"],
                "unit_id": uid,
                "train_r2": s["train_r2"],
                "cv_r2": s["cv_r2"],
                "cv_r2_worst_fold": float(np.min(s["cv_r2_per_fold"])),
                "delta_r2": s["delta_r2"],
                "n_iter": int(s["n_iter"]),
                "converged": bool(s["converged"]),
                "fit_s": time.perf_counter() - t0,
            })
            print(f"  [{v['name']}] {uid}: cv_r2={s['cv_r2']:+.4f} "
                  f"(worst fold {rows[-1]['cv_r2_worst_fold']:+.3f}) "
                  f"delta_r2={s['delta_r2']:+.4f}")

    return pl.DataFrame(rows)


def summarize(table: pl.DataFrame) -> pl.DataFrame:
    """Per-variant means over cells — the at-a-glance comparison."""
    return (
        table.group_by("variant")
        .agg(
            pl.col("cv_r2").mean().alias("mean_cv_r2"),
            pl.col("cv_r2").median().alias("median_cv_r2"),
            pl.col("cv_r2_worst_fold").min().alias("min_worst_fold"),
            pl.col("delta_r2").mean().alias("mean_delta_r2"),
            pl.col("converged").mean().alias("frac_converged"),
            pl.col("fit_s").mean().alias("mean_fit_s"),
        )
        .sort("mean_cv_r2", descending=True)
    )


if __name__ == "__main__":
    table = compare()
    print("\n=== per-fit ===")
    print(table)
    print("\n=== per-variant summary ===")
    print(summarize(table))
    print("\n=== cv_r2 by cell x variant ===")
    print(table.pivot(values="cv_r2", index="unit_id", on="variant"))

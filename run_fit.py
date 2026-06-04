"""Fit the bilinear gain GLM to every QC-pass unit in one session.

Run as a CLI — one invocation per session, which is what each SLURM job does
(see launch_slurm.py):

    python run_fit.py --nwb-path s3://.../668755_2023-08-31.nwb \
                      --session-id 668755_2023-08-31 \
                      --output-dir /path/to/results

It lazily loads the session's units table, keeps only `is_qc_pass` units, builds
the (session-shared) predictor design once, then fits each unit in parallel and
writes <output-dir>/<session_id>.json keyed by unit_id.
"""

import argparse
import json
import pathlib
import time
from typing import Any

import lazynwb
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from numpy import ndarray
from threadpoolctl import threadpool_limits

from bilinear_glm import (
    BilinearGLM,
    ContinuousPredictor,
    EventPredictor,
    GainModulator,
    bin_spike_times,
    make_trial_idx,
    windows_mask,
)

# so we don't need credentials to access the data on S3
lazynwb.config.anon = True

DT = 0.025

# Restrict the fit (and CV scoring) to peri-stimulus activity: this window,
# relative to each stimulus onset, defines the bins that are actually fit. The
# convolutional design is still built over the whole session (so kernels don't
# leak across the window edges) — only the regression target is masked.
STIM_FIT_WINDOW = (-0.5, 1.0)

# These cached NWBs flag QC-pass units with the boolean `default_qc` (there is no
# `is_qc_pass` column as in the open-data export). We additionally drop units the
# decoder labelled "noise".
QC_COLUMN = "default_qc"


def _to_jsonable(obj):
    """JSON-encode numpy arrays / scalars (json's default doesn't handle them)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    raise TypeError(f"unserializable type: {type(obj).__name__}")


def fit_one_unit(unit_id, y_unit, model, trial_idx, design, fit_mask=None):
    """Worker for the parallel pool: fits one cell and returns its summary.

    `threadpool_limits(1)` caps BLAS inside the worker — without this, each
    worker would spawn many BLAS threads and oversubscribe the cores (we
    measured this: 10 workers × 10 BLAS threads is much slower than 10
    workers × 1 BLAS thread for this workload).

    `fit_mask` (if given) restricts the fit and CV scoring to the marked bins
    (peri-stimulus windows); the design is still built over the full session.
    """
    with threadpool_limits(1):
        summary = model.fit_summary(
            y=y_unit,
            trial_idx=trial_idx,
            remove_gains=['context'],
            design=design,
            fit_mask=fit_mask,
            max_iter=50,
            tol=1e-3,
            verbose=False,
        )
    return unit_id, summary


def get_qc_pass_unit_ids(nwb_path: str, qc_column: str = QC_COLUMN) -> list[str]:
    """unit_ids of the session's QC-pass, non-noise units, in table order."""
    return (
        lazynwb.scan_nwb(nwb_path, "/units")
        .filter(pl.col(qc_column) & (pl.col("decoder_label") != "noise"))
        .select("unit_id")
        .collect()["unit_id"]
        .to_list()
    )


def load_y(nwb_path: str, unit_id: str, task_start_time: float,
           task_end_time: float, dt: float) -> ndarray:
    """Bin one unit's spike times into the same y-grid used for all fits."""
    spike_times = (
        lazynwb.scan_nwb(nwb_path, "/units")
        .filter(pl.col("unit_id") == unit_id)
        .select("spike_times")
        .collect()["spike_times"][0]
        .to_numpy()
    )
    return bin_spike_times(spike_times,
                           start=task_start_time + dt,
                           end=task_end_time + dt,
                           dt=dt)


def build_session_design(nwb_path: str, dt: float = DT) -> dict:
    """Build the predictors, model, trial index, and precomputed design for a
    session. Everything here is shared across all of the session's units, so it
    is built once and reused for every per-unit fit.

    Returns a dict with: model, trial_idx, design, task_start_time,
    task_end_time, T (number of time bins).
    """
    trials_df = lazynwb.read_nwb(nwb_path, "/intervals/trials")

    trial_start_times = trials_df.select("start_time").to_numpy().flatten()
    trial_end_times: ndarray[tuple[int], Any] = (
        trials_df.select("stop_time").to_numpy().flatten()
    )

    task_start_time = trial_start_times[0]
    task_end_time = trial_end_times[-1]

    # Number of time bins on the y-grid (same grid as load_y). Computed from the
    # grid rather than from a unit so the design can be sized without I/O.
    T = int(np.floor((task_end_time - task_start_time) / dt))

    trials_event_columns = {
        'is_aud_target': {'window_start': 0, 'window_end': 0.1, 'n_basis': 2},
        'is_aud_nontarget': {'window_start': 0, 'window_end': 0.1, 'n_basis': 2},
        'is_vis_target': {'window_start': 0, 'window_end': 0.1, 'n_basis': 2},
        'is_vis_nontarget': {'window_start': 0, 'window_end': 0.1, 'n_basis': 2},
        'is_hit': {'window_start': 0.1, 'window_end': 1, 'n_basis': 9},
        # 'is_miss': {'window_start': 0, 'window_end': 2, 'n_basis': 20},
        # 'is_correct_reject': {'window_start': 0, 'window_end': 2, 'n_basis': 20},
        # 'is_false_alarm': {'window_start': 0, 'window_end': 2, 'n_basis': 20},
    }
    predictors = []

    # add event predictors for each trial event type
    for col in trials_event_columns:
        event_times = trials_df.filter(pl.col(col)).select("stim_start_time")["stim_start_time"].to_numpy()
        window_start = trials_event_columns[col].get('window_start', 0)
        window_end = trials_event_columns[col].get('window_end', 1)
        n_basis = trials_event_columns[col].get('n_basis', 10)
        predictors.append(
            EventPredictor(col, event_times - task_start_time, window=(window_start, window_end), n_basis=n_basis, basis="cosine", gain_modulated=True)
        )

    # add licks
    lick_times = (
        lazynwb.scan_nwb(nwb_path, "/processing/behavior/licks")
        .select('timestamps')
        .collect()
        .to_numpy()
    )
    predictors.append(
        EventPredictor("licks", lick_times.flatten() - task_start_time, window=(0, 0.2), n_basis=5, basis="cosine", gain_modulated=True)
    )

    # add rewards
    reward_times = (
        lazynwb.scan_nwb(nwb_path, "/processing/behavior/rewards")
        .select('timestamps')
        .collect()
        .to_numpy()
    )
    predictors.append(
        EventPredictor("rewards", reward_times.flatten() - task_start_time, window=(-0.2, 1), n_basis=12, basis="cosine", gain_modulated=True)
    )

    # add continuous predictor for running speed
    running_speed = (
        lazynwb.scan_nwb(nwb_path, "/processing/behavior/running_speed")
        .select("timestamps", "data")
        .collect()
        .to_numpy()
    )
    predictors.append(
        ContinuousPredictor("running_speed",
        values=running_speed[:,1],
        window=(-1, 1),
        n_basis=10, basis="cosine",
        gain_modulated=False,
        times=running_speed[:, 0] - task_start_time,
        normalize='zscore'),
    )

    # add continuous predictor for pupil area
    pupil = (
        lazynwb.scan_nwb(nwb_path, "/processing/behavior/eye_tracking")
        .filter(~pl.col("pupil_is_bad_frame"))
        .select("timestamps", "pupil_area")
        .collect()
        .to_numpy()
    )
    predictors.append(
        ContinuousPredictor("pupil_area",
        values=pupil[:,1],
        window=(-1, 1),
        n_basis=10, basis="cosine",
        gain_modulated=False,
        times=pupil[:, 0] - task_start_time,
        normalize='zscore'),
    )

    # add continuous predictors for lightning pose features
    lp_features = {'ear': 'ear_base_l', 'jaw': 'jaw', 'nose': 'nose_tip', 'whisker_pad': 'whisker_pad_l_side'}

    lp = (
        lazynwb.scan_nwb(nwb_path, "/processing/behavior/lp_side_camera")
        .collect()
    )

    side_frame_times = (
        lazynwb.scan_nwb(nwb_path, "/acquisition/frametimes_side_camera")
        .select('timestamps')
        .collect()
        .to_numpy()
    )

    for feature_name, feature in lp_features.items():
        feature_x = lp.select(pl.col(feature + '_x')).to_numpy().flatten()
        feature_y = lp.select(pl.col(feature + '_y')).to_numpy().flatten()
        feature_likelihood = lp.select(pl.col(feature + "_likelihood")).to_numpy().flatten()
        feature_temporal_norm = lp.select(pl.col(feature + "_temporal_norm")).to_numpy().flatten()
        feature_times = side_frame_times.flatten()

        mask = (feature_likelihood > 0.98) & (feature_temporal_norm < np.nanmean(feature_temporal_norm) + 3*np.nanstd(feature_temporal_norm))
        feature_euclidean = np.sqrt(feature_x**2 + feature_y**2)

        predictors.append(
            ContinuousPredictor(feature_name,
            values=feature_euclidean[mask],
            window=(-1, 1),
            n_basis=10, basis="cosine",
            gain_modulated=False,
            times=feature_times[mask] - task_start_time,
            normalize='zscore'),
        )

    trial_context_labels = trials_df['is_vis_rewarded'].to_numpy().astype(int)*2 - 1  # -1 for aud trials, +1 for vis trials

    predictors.append(
        ContinuousPredictor("context_baseline",
        values=trial_context_labels,
        window=(0, 0),
        n_basis=1,
        basis="cosine",
        gain_modulated=False,
        times=trial_start_times - task_start_time,
        normalize='none'),
    )

    predictors.append(
        ContinuousPredictor("time",
        values=np.arange(T)/T,
        window=(0, 0),
        n_basis=1,
        basis="cosine",
        gain_modulated=False,
        normalize='zscore'),
    )

    predictors_with_gain = list(trials_event_columns.keys()) + ["licks", "rewards"]

    context_gain = trial_context_labels
    gains = [
        GainModulator("context", values=context_gain, modulates=predictors_with_gain),
    ]

    model = BilinearGLM(
        predictors=predictors,
        gains=gains,
        dt=dt,
        kernel_regularizer="ridge",      # ridge is fast for the demo; lasso also works
        alphas=np.logspace(-3, 3, 25),
        # spike_history = True
        # cv_folds=None (default): RidgeCV uses GCV/LOO via SVD — fast
        # cv_folds=5: k-fold CV — needed for LassoCV, slow for Ridge
    )

    trial_idx = make_trial_idx(
        trial_start_times - task_start_time,
        np.append(trial_start_times[1:], task_end_time) - task_start_time,
        dt,
    )

    # Precompute the design ONCE for the session — every cell shares the same
    # predictors / trial structure, so the per-cell convolutions are redundant.
    # fit_summary then skips that work for each cell.
    t0 = time.perf_counter()
    design = model.precompute_design(trial_idx, T=T)
    print(f"precompute_design took {time.perf_counter() - t0:.2f} s")

    # Restrict fitting/scoring to peri-stimulus windows. The mask marks bins
    # within STIM_FIT_WINDOW of any of the four stimulus onsets; the design is
    # still built over the full session above so kernels don't leak at edges.
    stim_columns = ['is_aud_target', 'is_aud_nontarget',
                    'is_vis_target', 'is_vis_nontarget']
    stim_onset_times = (
        trials_df.filter(pl.any_horizontal(pl.col(c) for c in stim_columns))
        .select("stim_start_time")["stim_start_time"]
        .drop_nulls()
        .to_numpy()
    )
    fit_mask = windows_mask(stim_onset_times - task_start_time, T, dt,
                            window=STIM_FIT_WINDOW)
    print(f"fit_mask keeps {fit_mask.sum()}/{T} bins "
          f"({100 * fit_mask.mean():.1f}%) within {STIM_FIT_WINDOW} s of "
          f"{len(stim_onset_times)} stimulus onsets")

    return {
        "model": model,
        "trial_idx": trial_idx,
        "design": design,
        "fit_mask": fit_mask,
        "task_start_time": task_start_time,
        "task_end_time": task_end_time,
        "T": T,
    }


def fit_session(nwb_path: str, session_id: str, output_dir: pathlib.Path, *,
                dt: float = DT, n_jobs: int = -1, limit: int | None = None,
                qc_column: str = QC_COLUMN, overwrite: bool = False) -> pathlib.Path:
    """Fit every QC-pass unit in one session and write its results JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{session_id}.json"
    if output_path.exists() and not overwrite:
        print(f"[{session_id}] {output_path} exists — skipping (use --overwrite to refit)")
        return output_path

    unit_ids = get_qc_pass_unit_ids(nwb_path, qc_column=qc_column)
    if limit is not None:
        unit_ids = unit_ids[:limit]
    print(f"[{session_id}] {len(unit_ids)} QC-pass units to fit")
    if not unit_ids:
        print(f"[{session_id}] no QC-pass units — nothing to do")
        return output_path

    sd = build_session_design(nwb_path, dt=dt)
    model, trial_idx, design = sd["model"], sd["trial_idx"], sd["design"]
    fit_mask = sd["fit_mask"]

    # Preload each unit's binned spike train serially (NWB / S3 I/O — concurrent
    # reads of the same file aren't always safe). Compute is what we parallelize.
    print(f"[{session_id}] loading spike trains for {len(unit_ids)} units...")
    t0 = time.perf_counter()
    y_by_unit = {
        uid: load_y(nwb_path, uid, sd["task_start_time"], sd["task_end_time"], dt)
        for uid in unit_ids
    }
    print(f"  loaded in {time.perf_counter() - t0:.2f} s")

    # Fit every cell in parallel, reusing the precomputed design. Each worker
    # sets BLAS to 1 thread to avoid oversubscription (see fit_one_unit).
    print(f"[{session_id}] fitting {len(unit_ids)} units with n_jobs={n_jobs}...")
    t_loop = time.perf_counter()
    parallel_out = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(fit_one_unit)(uid, y_by_unit[uid], model, trial_idx, design, fit_mask)
        for uid in unit_ids
    )
    results: dict[str, dict] = dict(parallel_out)
    print(f"\n[{session_id}] parallel fit of {len(results)} units took "
          f"{time.perf_counter() - t_loop:.2f} s")

    # Per-unit summary table
    for uid in unit_ids:
        if uid not in results:
            continue
        summary = results[uid]
        converged_tag = "" if summary['converged'] else "  [HIT max_iter]"
        print(f"{uid}: train_r2={summary['train_r2']:+.4f}  "
              f"cv_r2={summary['cv_r2']:+.4f}  "
              f"Δr²(context)={summary['delta_r2']:+.4f}  "
              f"iters={summary['n_iter']:>4d} "
              f"(folds={summary['n_iter_per_fold'].tolist()})"
              f"{converged_tag}")

    with open(output_path, "w") as f:
        json.dump(results, f, default=_to_jsonable, indent=2)
    print(f"\n[{session_id}] saved summaries to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Fit the bilinear gain GLM to all QC-pass units in one session.")
    parser.add_argument("--nwb-path", required=True,
                        help="Path/URL to the session NWB (passed to lazynwb).")
    parser.add_argument("--session-id", required=True,
                        help="Session id — used to name the output file.")
    parser.add_argument("--output-dir", required=True, type=pathlib.Path,
                        help="Directory to write <session_id>.json into.")
    parser.add_argument("--dt", type=float, default=DT,
                        help=f"Bin width in seconds (default {DT}).")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="joblib workers; -1 = all cores (default).")
    parser.add_argument("--qc-column", default=QC_COLUMN,
                        help=f"Boolean units column for QC-pass (default {QC_COLUMN}); "
                             "units labelled decoder noise are always excluded.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Fit only the first N units (for testing).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Refit even if the output file already exists.")
    args = parser.parse_args()

    fit_session(
        nwb_path=args.nwb_path,
        session_id=args.session_id,
        output_dir=args.output_dir,
        dt=args.dt,
        n_jobs=args.n_jobs,
        limit=args.limit,
        qc_column=args.qc_column,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

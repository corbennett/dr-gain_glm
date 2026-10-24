"""Session analyses and aggregation with explicit sampling units."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import rankdata

from .data import WINDOWS
from .statistics import (
    association_test,
    correlation,
    estimate_preferences,
    fdr,
    modality_null,
    mouse_summary,
    permutation_p,
    spatial_test,
    stable_seed,
)


def analyze_session(row, units, arrays, permutations, seed, minimum_trials=5):
    session_id = row["session_id"]
    summary, unit_rows = [], []
    coords = units.select("ccf_ap", "ccf_dv", "ccf_ml").to_numpy().astype(float)
    probes = units["electrode_group_name"].fill_null("unknown").to_numpy()
    areas = units["structure"].to_numpy()

    def add(area, window, metric, observed, null, *, two_sided=False, **extra):
        finite_null = np.isfinite(null).all()
        summary.append(
            {
                "session_id": session_id,
                "subject_id": str(row["subject_id"]),
                "cohort": row["cohort"],
                "structure": area,
                "window": window,
                "metric": metric,
                "estimate": observed,
                "null_mean": float(np.mean(null)) if finite_null else np.nan,
                "p_value": permutation_p(observed, null, two_sided=two_sided),
                "two_sided": two_sided,
                "n_cells": int(np.sum(areas == area)),
                "n_sites": None,
                "n_probes": None,
                "n_edges": None,
                "null": np.asarray(null, dtype=np.float32).tolist(),
                **extra,
            }
        )

    for window in WINDOWS:
        evoked = arrays[window].astype(float) - arrays["baseline"]
        estimates, means, _, scale, weights = estimate_preferences(
            arrays["baseline"],
            evoked,
            arrays["stimulus"],
            arrays["context"],
            arrays["block"],
            arrays["half"],
            arrays["nuisance"],
            minimum_trials,
        )
        null = modality_null(
            evoked,
            arrays["stimulus"],
            arrays["block"],
            arrays["half"],
            weights,
            permutations,
            stable_seed(seed, session_id, window, "modality"),
        )
        observed = estimates["modality"].mean(axis=1)
        full_null = null[:, :, :, 0].mean(axis=2)
        # All assignments, including observed, are ranked symmetrically. The
        # maximum rank for ties gives conservative randomization p-values.
        assignments = np.vstack([np.abs(observed), np.abs(full_null)])
        assignment_p = rankdata(-assignments, axis=0, method="max") / len(assignments)
        unit_p = assignment_p[0]
        for i, metadata in enumerate(units.iter_rows(named=True)):
            result = {
                **metadata,
                "window": window,
                "response_scale_hz": scale[i],
                "modality_p": unit_p[i],
                "n_trials": len(arrays["times"]),
                "n_contexts": len(np.unique(arrays["context"])),
                "visual_response_energy": estimates["visual_energy"][i],
                "auditory_response_energy": estimates["auditory_energy"][i],
            }
            for name, value in estimates.items():
                if value.ndim == 2:
                    result[f"{name}_h0"] = value[i, 0]
                    result[f"{name}_h1"] = value[i, 1]
                    result[f"{name}_hz"] = value[i].mean()
            for s, name in enumerate(
                (
                    "visual_target",
                    "visual_nontarget",
                    "auditory_target",
                    "auditory_nontarget",
                )
            ):
                result[f"{name}_response_hz"] = means[i, :, s].mean()
            # Positive is a stronger response magnitude to vision, including
            # suppression; negative crossvalidated energies are left unclipped.
            result["modality_energy_difference"] = (
                estimates["visual_energy"][i] - estimates["auditory_energy"][i]
            )
            unit_rows.append(result)

        for area in np.unique(areas):
            idx = np.flatnonzero(areas == area)
            if len(idx) < 10:
                continue
            scaled = {
                name: values[idx] / scale[idx, None]
                for name, values in estimates.items()
                if values.ndim == 2
            }
            n = null[:, idx] / scale[None, idx, None, None]
            m = scaled["modality"]
            add(
                area,
                window,
                "modality_power",
                np.mean(m[:, 0] * m[:, 1]),
                np.mean(n[:, :, 0, 0] * n[:, :, 1, 0], axis=1),
            )
            mt, mn = scaled["modality_target"], scaled["modality_nontarget"]
            add(
                area,
                window,
                "modality_generalization",
                np.mean(mt * mn[:, ::-1]),
                np.mean(n[:, :, :, 1] * n[:, :, ::-1, 2], axis=(1, 2)),
            )
            add(
                area,
                window,
                "modality_selective_fraction",
                np.mean(unit_p[idx] <= 0.05),
                np.mean(assignment_p[1:, idx] <= 0.05, axis=1),
            )
            add(
                area,
                window,
                "visual_bias",
                m.mean(),
                n[:, :, :, 0].mean(axis=(1, 2)),
                two_sided=True,
            )
            # Reliability is descriptive here; power and generalization have
            # the prespecified stimulus-label null tests above.
            rho = correlation(
                estimates["modality"][idx, 0], estimates["modality"][idx, 1]
            )
            add(
                area,
                window,
                "modality_split_reliability",
                rho,
                np.array([correlation(a[:, 0, 0], a[:, 1, 0]) for a in null[:, idx]]),
            )
            pairs = {
                "baseline_modality": ("baseline", "modality", False),
                "baseline_modality_within_probe": ("baseline", "modality", True),
                "baseline_raw_modality": ("baseline_raw", "modality", False),
                "baseline_target": ("baseline", "target", False),
                "baseline_target_visual": ("baseline", "target_visual", False),
                "baseline_target_auditory": ("baseline", "target_auditory", False),
            }
            for metric, (a, b, within_probe) in pairs.items():
                value, shuffled = association_test(
                    estimates[a][idx],
                    estimates[b][idx],
                    permutations,
                    stable_seed(seed, session_id, area, window, metric),
                    strata=probes[idx] if within_probe else None,
                )
                add(area, window, metric, value, shuffled, two_sided=True)

            baseline = estimates["baseline"][idx]
            delta_target = scaled["target_visual"] - scaled["target_auditory"]
            # Select preferred context using one half, compare target preference
            # in matching versus opposite modality using the other half.
            value = np.mean(np.sign(baseline) * delta_target[:, ::-1])
            rng = np.random.default_rng(
                stable_seed(seed, session_id, area, window, "target_advantage")
            )
            shuffled = np.array(
                [
                    np.mean(
                        np.sign(baseline[rng.permutation(len(idx))])
                        * delta_target[:, ::-1]
                    )
                    for _ in range(permutations)
                ]
            )
            add(
                area,
                window,
                "preferred_modality_target_advantage",
                value,
                shuffled,
                two_sided=True,
            )

            spatial_metrics = {"modality": estimates["modality"][idx]}
            if window == "early":
                spatial_metrics["baseline"] = estimates["baseline"][idx]
            for name, values in spatial_metrics.items():
                spatial_window = "baseline" if name == "baseline" else window
                for within_probe in (False, True):
                    metric = f"spatial_{name}" + (
                        "_within_probe" if within_probe else ""
                    )
                    value, shuffled, meta = spatial_test(
                        values,
                        coords[idx],
                        probes[idx],
                        permutations,
                        stable_seed(seed, session_id, area, spatial_window, metric),
                        within_probe=within_probe,
                    )
                    add(area, spatial_window, metric, value, shuffled, **meta)

    return pl.from_dicts(unit_rows, infer_schema_length=None), pl.from_dicts(
        summary, infer_schema_length=None
    )


def aggregate(session_stats, unit_stats, output_dir, seed):
    """Write area/cohort inference, session inference, and cohort comparisons."""
    from pathlib import Path

    output_dir = Path(output_dir)
    summary = []
    keys = ["structure", "cohort", "window", "metric"]
    for group, frame in session_stats.group_by(keys, maintain_order=True):
        frame = frame.sort("session_id")
        values = frame["estimate"].to_numpy()
        nulls = np.asarray(frame["null"].to_list())
        result = mouse_summary(
            values,
            frame["subject_id"].to_numpy(),
            nulls,
            stable_seed(seed, *group, "bootstrap"),
        )
        if result is None:
            continue
        result.update(dict(zip(keys, group)))
        result["n_cells"] = frame.filter(pl.col("estimate").is_finite())[
            "n_cells"
        ].sum()
        two_sided = bool(frame["two_sided"][0])
        result["p_value"] = result["p_two_sided"] if two_sided else result["p_upper"]
        summary.append(result)
    if not summary:
        raise ValueError("No analyzable session/area groups")
    area = pl.from_dicts(summary).sort(*keys)
    # One explicit family: all finite area × cohort × window × metric hypotheses.
    area = area.with_columns(pl.Series("q_value", fdr(area["p_value"].to_numpy())))
    area.write_csv(output_dir / "area_summary.csv")
    session_stats = session_stats.with_columns(
        pl.Series("q_value", fdr(session_stats["p_value"].to_numpy()))
    )
    session_stats.drop("null").write_csv(output_dir / "session_statistics.csv")
    session_stats.write_parquet(output_dir / "session_statistics.parquet")
    # Unit FDR is per area × cohort × window, separate from population inference.
    unit_frames = []
    for _, frame in unit_stats.group_by("structure", "cohort", "window"):
        q = fdr(frame["modality_p"].to_numpy())
        unit_frames.append(frame.with_columns(pl.Series("modality_q", q)))
    unit_stats = pl.concat(unit_frames).sort("session_id", "unit_id", "window")
    unit_stats.write_parquet(output_dir / "unit_preferences.parquet")
    comparisons = []
    for group, frame in area.group_by("structure", "window", "metric"):
        by_cohort = {r["cohort"]: r for r in frame.iter_rows(named=True)}
        if "expert" not in by_cohort or "naive" not in by_cohort:
            continue
        e, n = by_cohort["expert"], by_cohort["naive"]
        record = {
            **dict(zip(("structure", "window", "metric"), group)),
            "expert_estimate": e["estimate"],
            "naive_estimate": n["estimate"],
            "naive_minus_expert": n["estimate"] - e["estimate"],
            "expert_mice": e["n_mice"],
            "naive_mice": n["n_mice"],
            "same_sign": np.sign(e["estimate"]) == np.sign(n["estimate"]),
            "difference_ci_low": np.nan,
            "difference_ci_high": np.nan,
            "population_comparison": e["n_mice"] >= 3 and n["n_mice"] >= 3,
        }
        if record["population_comparison"]:
            sub = session_stats.filter(
                *[
                    pl.col(k) == v
                    for k, v in zip(("structure", "window", "metric"), group)
                ]
            )
            rng = np.random.default_rng(stable_seed(seed, *group, "cohort_difference"))
            samples = {}
            for cohort in ("expert", "naive"):
                means = (
                    sub.filter(
                        pl.col("cohort") == cohort, pl.col("estimate").is_finite()
                    )
                    .group_by("subject_id")
                    .agg(pl.col("estimate").mean())["estimate"]
                    .to_numpy()
                )
                samples[cohort] = means[
                    rng.integers(len(means), size=(2000, len(means)))
                ].mean(axis=1)
            interval = np.quantile(samples["naive"] - samples["expert"], [0.025, 0.975])
            record["difference_ci_low"], record["difference_ci_high"] = interval
        comparisons.append(record)
    if comparisons:
        pl.from_dicts(comparisons).sort("structure", "window", "metric").write_csv(
            output_dir / "cohort_comparison.csv"
        )
    return area, session_stats, unit_stats

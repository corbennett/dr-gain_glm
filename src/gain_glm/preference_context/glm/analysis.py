"""Notebook-compatible GLM population labels, coverage, and mouse comparisons."""

from __future__ import annotations

import numpy as np
import polars as pl

from gain_glm.context_preferences.statistics import fdr, stable_seed
from gain_glm.preference_context.statistics import (
    match_groups,
    partial_rank_correlation,
    preference_categories,
)
from gain_glm.responsive_preferences.statistics import adjust_by_group, mouse_inference

GLM_COLUMNS = {
    "context_baseline_dropout_permutation_p_value": "glm_baseline_p",
    "context_baseline_dropout_permutation_observed_delta_r2": "glm_delta_r2",
    "context_baseline_dropout_delta_r2": "glm_unshifted_delta_r2",
    "context_baseline_dropout_permutation_n_unique_shifts": "glm_n_shifts",
    "context_baseline_dropout_metric_converged": "glm_baseline_converged",
    "converged": "glm_fit_converged",
    "cv_converged": "glm_cv_converged",
    "cv_r2": "glm_cv_r2",
}
GROUP = ["structure", "cohort", "window", "selection_half", "fit_subset"]
METRICS = ("glm_nominal_probability", "glm_fdr_probability", "glm_delta_r2")


def link_data(metadata, preferences, baseline_covariates, glm):
    """FDR uses all eligible QC cells, not preference-specific subsets."""
    if glm.select("session_id", "unit_id").is_duplicated().any():
        raise ValueError("Duplicate GLM cell identities")
    glm = glm.rename(GLM_COLUMNS)
    universe = metadata.join(
        glm, on=["session_id", "unit_id"], how="left", validate="1:1"
    )
    universe = universe.with_columns(
        pl.col("glm_baseline_p").is_finite().fill_null(False).alias("glm_available"),
        (
            pl.col("glm_fit_converged").fill_null(False)
            & pl.col("glm_cv_converged").fill_null(False)
            & pl.col("glm_baseline_converged").fill_null(False)
        ).alias("glm_converged"),
    )
    finite = universe.filter(pl.col("glm_available"))
    if finite.filter(
        (pl.col("glm_baseline_p") < 0) | (pl.col("glm_baseline_p") > 1)
    ).height:
        raise ValueError("Invalid GLM p-value")
    adjusted = []
    for _, f in finite.group_by("structure", "cohort", maintain_order=True):
        adjusted.append(
            f.with_columns(
                pl.Series("glm_baseline_q", fdr(f["glm_baseline_p"].to_numpy()))
            )
        )
    if not adjusted:
        raise ValueError("No GLM fits match the selected dataset")
    finite = pl.concat(adjusted).with_columns(
        (pl.col("glm_baseline_p") < 0.05)
        .cast(pl.Float64)
        .alias("glm_nominal_probability"),
        (pl.col("glm_baseline_q") < 0.05).cast(pl.Float64).alias("glm_fdr_probability"),
    )
    coverage = universe.group_by("session_id", "subject_id", "structure", "cohort").agg(
        pl.len().alias("n_qc"),
        pl.col("glm_available").sum().alias("n_glm"),
        pl.col("glm_converged").sum().alias("n_converged"),
    )
    columns = [
        "session_id",
        "unit_id",
        *GLM_COLUMNS.values(),
        "glm_converged",
        "glm_baseline_q",
        *METRICS[:2],
    ]
    joined = preferences.join(
        finite.select(columns),
        on=["session_id", "unit_id"],
        how="inner",
        validate="m:1",
    )
    covariates = baseline_covariates.filter(pl.col("preference_quarter") == 0).select(
        "session_id", "unit_id", "window", "selection_half", "selection_baseline_hz"
    )
    joined = joined.join(
        covariates,
        on=["session_id", "unit_id", "window", "selection_half"],
        how="left",
        validate="1:1",
    )
    if joined["selection_baseline_hz"].null_count():
        raise ValueError("Missing selection-half baseline covariates")
    exclusive, dominant = preference_categories(
        joined["modality_q"].to_numpy(),
        joined["target_q"].to_numpy(),
        joined["modality_hz"].to_numpy(),
        joined["target_hz"].to_numpy(),
    )
    joined = joined.with_columns(
        pl.Series("exclusive_class", exclusive), pl.Series("dominant_class", dominant)
    )
    return joined, coverage, finite


def compare(units, output, seed, minimum=3, caliper=1.5):
    units.write_parquet(output / "linked_neurons.parquet")
    classes, comparisons, audit, matches = [], [], [], []
    session_keys = ["session_id", "subject_id", *GROUP[:-1]]
    for key, original in units.group_by(session_keys, maintain_order=True):
        for subset in ("all", "converged"):
            f = (
                original
                if subset == "all"
                else original.filter(pl.col("glm_converged"))
            )
            if not f.height:
                continue
            meta = {**dict(zip(session_keys, key)), "fit_subset": subset}
            m, t = f["modality_hz"].to_numpy(), f["target_hz"].to_numpy()
            features = np.log1p(
                np.column_stack(
                    [
                        f["selection_baseline_hz"],
                        f["selection_response_rms_hz"],
                        np.hypot(m, t),
                    ]
                )
            )
            balance = (m * m - t * t) / (m * m + t * t + 1e-12)
            rho = partial_rank_correlation(
                balance, f["glm_delta_r2"].to_numpy(), features
            )
            if np.isfinite(rho):
                comparisons.append(
                    {
                        **meta,
                        "comparison": "continuous",
                        "adjustment": "partial_rank",
                        "metric": "glm_partial_correlation",
                        "estimate": rho,
                        "left": np.nan,
                        "right": np.nan,
                        "n_left": f.height,
                        "n_right": f.height,
                    }
                )
            for scheme, a, b in (
                ("exclusive", "modality_only", "target_only"),
                ("dominant", "modality_dominant", "target_dominant"),
            ):
                labels = f[f"{scheme}_class"].to_numpy()
                for label in np.unique(labels):
                    g = f.filter(pl.col(f"{scheme}_class") == label)
                    classes.append(
                        {
                            **meta,
                            "scheme": scheme,
                            "category": label,
                            "n_cells": g.height,
                            **{metric: g[metric].mean() for metric in METRICS},
                        }
                    )
                left, right = np.flatnonzero(labels == a), np.flatnonzero(labels == b)
                pairs, distances, z = match_groups(features, left, right, caliper)
                audit.append(
                    {
                        **meta,
                        "comparison": scheme,
                        "n_left": len(left),
                        "n_right": len(right),
                        "n_matched_pairs": len(pairs),
                        "eligible_unmatched": min(len(left), len(right)) >= minimum,
                        "eligible_matched": len(pairs) >= minimum,
                    }
                )
                for (i, j), distance in zip(pairs, distances):
                    matches.append(
                        {
                            **meta,
                            "comparison": scheme,
                            "modality_unit_id": f["unit_id"][int(i)],
                            "target_unit_id": f["unit_id"][int(j)],
                            "distance": distance,
                            **{
                                f"balance_{name}": float(z[i, k] - z[j, k])
                                for k, name in enumerate(
                                    ("rate", "response", "preference")
                                )
                            },
                        }
                    )
                for adjustment, i, j in (
                    ("unmatched", left, right),
                    ("matched", pairs[:, 0], pairs[:, 1]),
                ):
                    if min(len(i), len(j)) < minimum:
                        continue
                    for metric in METRICS:
                        values = f[metric].to_numpy()
                        a_mean, b_mean = (
                            float(values[i].mean()),
                            float(values[j].mean()),
                        )
                        comparisons.append(
                            {
                                **meta,
                                "comparison": scheme,
                                "adjustment": adjustment,
                                "metric": metric,
                                "estimate": a_mean - b_mean,
                                "left": a_mean,
                                "right": b_mean,
                                "n_left": len(i),
                                "n_right": len(j),
                            }
                        )
    pl.from_dicts(classes).write_csv(output / "session_classes.csv")
    pl.from_dicts(audit).write_csv(output / "comparison_support.csv")
    if matches:
        pl.from_dicts(matches).write_csv(output / "matched_pairs.csv")
    sessions = pl.from_dicts(comparisons, infer_schema_length=None)
    sessions.write_csv(output / "session_comparisons.csv")
    keys = [*GROUP, "comparison", "adjustment", "metric"]
    summaries = []
    for key, f in sessions.group_by(keys, maintain_order=True):
        stats = mouse_inference(
            f["estimate"].to_numpy(),
            f["subject_id"].to_numpy(),
            stable_seed(seed, *key),
        )
        if key[-1].endswith("probability"):
            values = f.select("left", "right").to_numpy()
            if np.all(values == 0) or np.all(values == 1):
                stats.update(
                    ci_low=np.nan,
                    ci_high=np.nan,
                    p_value=np.nan,
                    test="no_outcome_variation",
                    population_inference=False,
                )
        means = f.group_by("subject_id").agg(
            pl.col("left").mean(), pl.col("right").mean()
        )
        summaries.append(
            {
                **dict(zip(keys, key)),
                **stats,
                "left": means["left"].mean(),
                "right": means["right"].mean(),
                "n_left": f["n_left"].sum(),
                "n_right": f["n_right"].sum(),
            }
        )
    summary = adjust_by_group(
        pl.from_dicts(summaries),
        ["window", "selection_half", "fit_subset", "adjustment", "metric"],
        "p_value",
        "q_value",
    ).sort(*keys)
    summary.write_csv(output / "area_summary.csv")
    mice = sessions.group_by(*keys, "subject_id").agg(
        pl.col("estimate").mean(),
        pl.col("left").mean(),
        pl.col("right").mean(),
        pl.len().alias("n_sessions"),
        pl.col("n_left").sum(),
        pl.col("n_right").sum(),
    )
    mice.write_csv(output / "mouse_estimates.csv")
    units.group_by(
        "structure",
        "cohort",
        "window",
        "selection_half",
        "exclusive_class",
        "dominant_class",
    ).agg(
        pl.len().alias("n_cells"),
        pl.col("glm_nominal_probability").sum().alias("n_context_nominal"),
        pl.col("glm_fdr_probability").sum().alias("n_context_fdr"),
        pl.col("glm_converged").sum().alias("n_converged"),
    ).write_csv(output / "class_counts.csv")
    return summary

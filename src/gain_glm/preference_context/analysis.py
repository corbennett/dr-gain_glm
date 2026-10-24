"""Separate responsive-cell selection, preference classification, and baseline."""

from __future__ import annotations

import numpy as np
import polars as pl

from gain_glm.context_preferences.statistics import fdr, stable_seed
from gain_glm.responsive_preferences.analysis import validate_arrays
from gain_glm.responsive_preferences.statistics import (
    FACTOR_WEIGHTS,
    FACTORS,
    adjust_by_group,
    evaluation_design,
    factor_null,
    mouse_inference,
    unit_pvalues,
)

from .statistics import (
    block_pair_power,
    cluster_context_test,
    match_groups,
    partial_rank_correlation,
    preference_categories,
    selection_baseline_scale,
)

GROUP = ["structure", "cohort", "window", "selection_half", "preference_quarter"]
SESSION_GROUP = ["session_id", "subject_id", *GROUP]
METRICS = ("encoding_probability", "linear_encoding_probability", "block_pair_power")


def evaluate_session(selection, units, arrays, seed, permutations):
    validate_arrays(units, arrays)
    sid = units["session_id"][0]
    lookup = {u: i for i, u in enumerate(units["unit_id"])}
    frames = []
    for selection_half in (0, 1):
        keep, stimulus, block, split, weights = evaluation_design(
            arrays, selection_half, seed, sid
        )
        context, nuisance = arrays["context"][keep], arrays["nuisance"][keep]
        for window in ("early", "late"):
            selected = selection.filter(
                pl.col("selection_half") == selection_half, pl.col("window") == window
            ).sort("unit_id")
            if not selected.height:
                continue
            idx = np.array([lookup[u] for u in selected["unit_id"]])
            baseline = arrays["baseline"][idx].astype(float)
            evoked = (arrays[window][idx].astype(float) - baseline)[:, keep]
            contrasts = np.einsum("ut,hst->uhs", evoked, weights) @ FACTOR_WEIGHTS.T
            nulls = [
                factor_null(
                    evoked,
                    stimulus,
                    block,
                    split,
                    weights,
                    f,
                    permutations,
                    stable_seed(seed, sid, selection_half, window, f),
                )
                for f in FACTORS
            ]
            scale = selection_baseline_scale(baseline[:, ~keep], arrays["block"][~keep])
            selection_rate = baseline[:, ~keep].mean(axis=1)
            for quarter in (0, 1):
                test = split != quarter
                base = baseline[:, keep][:, test]
                cubic = cluster_context_test(
                    base, context[test], nuisance[test], block[test]
                )
                linear = cluster_context_test(
                    base, context[test], nuisance[test], block[test], cubic=False
                )
                power, pair_coefficients = block_pair_power(
                    base, context[test], nuisance[test], block[test], scale
                )
                frame = selected.select(
                    "unit_id",
                    "session_id",
                    "subject_id",
                    "structure",
                    "cohort",
                    "window",
                    "selection_half",
                    "response_q",
                    "selection_response_rms_hz",
                ).with_columns(
                    pl.lit(quarter).alias("preference_quarter"),
                    pl.lit(int((~keep).sum())).alias("selection_trials"),
                    pl.lit(int((split == quarter).sum())).alias("preference_trials"),
                    pl.lit(int(test.sum())).alias("baseline_trials"),
                    pl.Series("selection_baseline_hz", selection_rate),
                    pl.Series("baseline_scale_hz", scale),
                    pl.Series("baseline_hz", cubic["hz"]),
                    pl.Series("baseline_se_hz", cubic["se_hz"]),
                    pl.Series("baseline_p", cubic["p"]),
                    pl.lit(cubic["df"]).alias("baseline_df"),
                    pl.lit(cubic["n_blocks"]).alias("n_blocks"),
                    pl.Series("baseline_linear_p", linear["p"]),
                    pl.Series("baseline_linear_hz", linear["hz"]),
                    pl.Series("block_pair_power", power),
                    pl.Series("block_pair_coefficients_hz", pair_coefficients.tolist()),
                )
                for j, factor in enumerate(FACTORS):
                    observed = contrasts[:, quarter, j]
                    frame = frame.with_columns(
                        pl.Series(f"{factor}_hz", observed),
                        pl.Series(
                            f"{factor}_p",
                            unit_pvalues(observed, nulls[j][:, :, quarter]),
                        ),
                    )
                frames.append(frame)
    return pl.concat(frames) if frames else None


def classify(frame):
    adjusted = []
    for _, f in frame.group_by(GROUP, maintain_order=True):
        qs = fdr(f.select("modality_p", "target_p").to_numpy().ravel()).reshape(-1, 2)
        exclusive, dominant = preference_categories(
            qs[:, 0], qs[:, 1], f["modality_hz"].to_numpy(), f["target_hz"].to_numpy()
        )
        bq = fdr(f["baseline_p"].to_numpy())
        lq = fdr(f["baseline_linear_p"].to_numpy())
        adjusted.append(
            f.with_columns(
                pl.Series("modality_q", qs[:, 0]),
                pl.Series("target_q", qs[:, 1]),
                pl.Series("baseline_q", bq),
                pl.Series("baseline_linear_q", lq),
                pl.Series("encoding_probability", (bq < 0.05).astype(float)),
                pl.Series("linear_encoding_probability", (lq < 0.05).astype(float)),
                pl.Series("exclusive_class", exclusive),
                pl.Series("dominant_class", dominant),
            )
        )
    return pl.concat(adjusted).sort(*SESSION_GROUP, "unit_id")


def aggregate(units, output, seed, minimum=3, caliper=1.5):
    """Compare groups only within a session, then give each mouse equal weight."""
    units = classify(units)
    units.write_parquet(output / "selected_neurons.parquet")
    class_rows, comparisons, audit, matches = [], [], [], []
    for key, f in units.group_by(SESSION_GROUP, maintain_order=True):
        meta = dict(zip(SESSION_GROUP, key))
        features = np.log1p(
            np.column_stack(
                [
                    f["selection_baseline_hz"],
                    f["selection_response_rms_hz"],
                    np.hypot(f["modality_hz"], f["target_hz"]),
                ]
            )
        )
        m, t = f["modality_hz"].to_numpy(), f["target_hz"].to_numpy()
        balance = (m * m - t * t) / (m * m + t * t + 1e-12)
        rho = partial_rank_correlation(
            balance, f["block_pair_power"].to_numpy(), features
        )
        if np.isfinite(rho):
            comparisons.append(
                {
                    **meta,
                    "comparison": "continuous",
                    "adjustment": "partial_rank",
                    "metric": "block_pair_correlation",
                    "estimate": rho,
                    "left": np.nan,
                    "right": np.nan,
                    "n_left": f.height,
                    "n_right": f.height,
                }
            )
        for scheme, left_name, right_name in (
            ("exclusive", "modality_only", "target_only"),
            ("dominant", "modality_dominant", "target_dominant"),
        ):
            labels = f[f"{scheme}_class"].to_numpy()
            for label in np.unique(labels):
                group = f.filter(pl.col(f"{scheme}_class") == label)
                class_rows.append(
                    {
                        **meta,
                        "scheme": scheme,
                        "category": label,
                        "n_cells": group.height,
                        **{metric: group[metric].mean() for metric in METRICS},
                    }
                )
            left, right = (
                np.flatnonzero(labels == left_name),
                np.flatnonzero(labels == right_name),
            )
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
            for pair, distance in zip(pairs, distances):
                matches.append(
                    {
                        **meta,
                        "comparison": scheme,
                        "modality_unit_id": f["unit_id"][int(pair[0])],
                        "target_unit_id": f["unit_id"][int(pair[1])],
                        "distance": distance,
                        **{
                            f"balance_{name}": float(z[pair[0], j] - z[pair[1], j])
                            for j, name in enumerate(("rate", "response", "preference"))
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
                    a, b = float(values[i].mean()), float(values[j].mean())
                    comparisons.append(
                        {
                            **meta,
                            "comparison": scheme,
                            "adjustment": adjustment,
                            "metric": metric,
                            "estimate": a - b,
                            "left": a,
                            "right": b,
                            "n_left": len(i),
                            "n_right": len(j),
                        }
                    )
    classes = pl.from_dicts(class_rows)
    classes.write_csv(output / "session_classes.csv")
    pl.from_dicts(audit).write_csv(output / "comparison_support.csv")
    if matches:
        pl.from_dicts(matches).write_csv(output / "matched_pairs.csv")
    sessions = pl.from_dicts(comparisons, infer_schema_length=None)
    sessions.write_csv(output / "session_comparisons.csv")
    keys = [*GROUP, "comparison", "adjustment", "metric"]
    summaries = []
    for key, f in sessions.group_by(keys, maintain_order=True):
        inference = mouse_inference(
            f["estimate"].to_numpy(),
            f["subject_id"].to_numpy(),
            stable_seed(seed, *key),
        )
        if key[-1].endswith("encoding_probability"):
            outcomes = f.select("left", "right").to_numpy()
            if np.all(outcomes == 0) or np.all(outcomes == 1):
                # An all-zero detection assay cannot support a precise equality
                # claim. A nonparametric bootstrap would misleadingly give [0,0].
                inference.update(
                    ci_low=np.nan,
                    ci_high=np.nan,
                    p_value=np.nan,
                    test="no_outcome_variation",
                    population_inference=False,
                )
        # Same mice and sessions for left/right percentages and their difference.
        means = f.group_by("subject_id").agg(
            pl.col("left").mean(), pl.col("right").mean()
        )
        summaries.append(
            {
                **dict(zip(keys, key)),
                **inference,
                "left": means["left"].mean(),
                "right": means["right"].mean(),
                "n_left": f["n_left"].sum(),
                "n_right": f["n_right"].sum(),
            }
        )
    summary = pl.from_dicts(summaries)
    # Separate prespecified families for each window, split, adjustment and outcome;
    # each jointly includes both categorical comparisons and all areas/cohorts.
    summary = adjust_by_group(
        summary,
        [
            "window",
            "selection_half",
            "preference_quarter",
            "adjustment",
            "metric",
        ],
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
    counts = units.group_by(*GROUP, "exclusive_class", "dominant_class").agg(
        pl.len().alias("n_cells"),
        pl.col("encoding_probability").sum().alias("n_context"),
        pl.col("session_id").n_unique().alias("n_sessions"),
        pl.col("subject_id").n_unique().alias("n_mice"),
    )
    counts.write_csv(output / "class_counts.csv")
    return units, summary, mice, classes

"""Two-stage analysis using cached trial rates from the 17-area dataset."""

from __future__ import annotations

import numpy as np
import polars as pl

from gain_glm.context_preferences.statistics import fdr, stable_seed

from .statistics import (
    FACTORS,
    adjust_by_group,
    evaluation_design,
    factor_null,
    mouse_inference,
    preference_estimates,
    response_selection,
    unit_pvalues,
)

GROUP = ["structure", "cohort", "window", "selection_half"]
SESSION_GROUP = ["session_id", "subject_id", *GROUP]
POWER_METRICS = ("modality_power", "target_power")
DIFFERENCE_METRICS = ("power_difference", "power_difference_hz2")


def validate_arrays(units, arrays):
    if units["unit_id"].to_list() != arrays["unit_ids"].tolist():
        raise ValueError("Unit order does not match cached rate arrays")


def prepare_session(units, arrays, seed):
    validate_arrays(units, arrays)
    sid = units["session_id"][0]
    rows = []
    for selection_half in (0, 1):
        selection_mask = arrays["half"] == selection_half
        keep, _, _, _, weights = evaluation_design(arrays, selection_half, seed, sid)
        for window in ("early", "late"):
            evoked = arrays[window].astype(float) - arrays["baseline"]
            response = response_selection(
                evoked, arrays["stimulus"], arrays["context"], selection_mask
            )
            estimates = preference_estimates(
                evoked[:, keep], weights, response["normalization_hz"]
            )
            for i, metadata in enumerate(units.iter_rows(named=True)):
                result = {
                    **metadata,
                    "window": window,
                    "selection_half": selection_half,
                    "selection_trials": int(selection_mask.sum()),
                    "evaluation_trials": int(keep.sum()),
                    "response_p": response["response_p"][i],
                    "selection_response_rms_hz": response["selection_response_rms_hz"][
                        i
                    ],
                    "normalization_hz": response["normalization_hz"][i],
                    "response_condition_means": response["response_condition_means"][
                        i
                    ].tolist(),
                    "response_condition_p": response["response_condition_p"][
                        i
                    ].tolist(),
                    "power_difference": estimates["power_difference"][i],
                    "power_difference_hz2": estimates["power_difference_hz2"][i],
                }
                for j, factor in enumerate(FACTORS):
                    result[f"{factor}_a_hz"] = estimates["contrasts"][i, 0, j]
                    result[f"{factor}_b_hz"] = estimates["contrasts"][i, 1, j]
                    result[f"{factor}_hz"] = estimates["contrasts"][i, :, j].mean()
                    result[f"{factor}_scaled"] = estimates["scaled"][i, :, j].mean()
                    result[f"{factor}_power"] = estimates["powers"][i, j]
                for j, name in enumerate(
                    (
                        "visual_target",
                        "visual_nontarget",
                        "auditory_target",
                        "auditory_nontarget",
                    )
                ):
                    result[f"{name}_response_hz"] = estimates["means"][i, :, j].mean()
                rows.append(result)
    return pl.from_dicts(rows, infer_schema_length=None)


def select_responsive(frame, alpha):
    frame = adjust_by_group(frame, GROUP, "response_p", "response_q")
    return frame.with_columns((pl.col("response_q") <= alpha).alias("responsive"))


def evaluate_session(selection, units, arrays, seed, permutations):
    validate_arrays(units, arrays)
    sid = units["session_id"][0]
    id_to_index = {u: i for i, u in enumerate(units["unit_id"])}
    unit_results, session_results = [], []
    for selection_half in (0, 1):
        keep, stimulus, block, split, weights = evaluation_design(
            arrays, selection_half, seed, sid
        )
        for window in ("early", "late"):
            selected = selection.filter(
                pl.col("responsive"),
                pl.col("selection_half") == selection_half,
                pl.col("window") == window,
            ).sort("unit_id")
            if not selected.height:
                continue
            idx = np.array([id_to_index[u] for u in selected["unit_id"]])
            evoked = (arrays[window][idx].astype(float) - arrays["baseline"][idx])[
                :, keep
            ]
            normalization = selected["normalization_hz"].to_numpy()
            area_ids = selected["structure"].to_numpy()
            new_columns = []
            for factor in FACTORS:
                null = factor_null(
                    evoked,
                    stimulus,
                    block,
                    split,
                    weights,
                    factor,
                    permutations,
                    stable_seed(seed, sid, selection_half, window, factor),
                )
                new_columns.append(
                    pl.Series(
                        f"{factor}_p",
                        unit_pvalues(
                            selected[f"{factor}_hz"].to_numpy(), null.mean(axis=2)
                        ),
                    )
                )
                null_power = null[:, :, 0] * null[:, :, 1] / normalization[None] ** 2
                for area in np.unique(area_ids):
                    positions = np.flatnonzero(area_ids == area)
                    first = selected.row(int(positions[0]), named=True)
                    session_results.append(
                        {
                            **{k: first[k] for k in SESSION_GROUP},
                            "metric": f"{factor}_power",
                            "estimate": float(
                                selected[f"{factor}_power"].to_numpy()[positions].mean()
                            ),
                            "n_selected": len(positions),
                            "null": null_power[:, positions].mean(axis=1).tolist(),
                        }
                    )
            unit_results.append(selected.with_columns(new_columns))
    return (
        pl.concat(unit_results) if unit_results else None,
        pl.from_dicts(session_results, infer_schema_length=None)
        if session_results
        else None,
    )


def aggregate(selection, units, sessions, output, seed):
    """Matched selected cells for both factors; retain both split orientations."""
    # Both factor tests enter one unit-level family, per area/cohort/window/split.
    adjusted = []
    for _, frame in units.group_by(GROUP, maintain_order=True):
        ps = frame.select("modality_p", "target_p").to_numpy()
        qs = fdr(ps.ravel()).reshape(ps.shape)
        adjusted.append(
            frame.with_columns(
                pl.Series("modality_q", qs[:, 0]), pl.Series("target_q", qs[:, 1])
            )
        )
    units = pl.concat(adjusted).sort(
        "session_id", "unit_id", "window", "selection_half"
    )
    units.write_parquet(output / "selected_neurons.parquet")
    counts = (
        selection.group_by(GROUP)
        .agg(
            pl.len().alias("n_tested"),
            pl.col("responsive").sum().alias("n_selected"),
            pl.col("session_id")
            .filter(pl.col("responsive"))
            .n_unique()
            .alias("selected_sessions"),
            pl.col("subject_id")
            .filter(pl.col("responsive"))
            .n_unique()
            .alias("selected_mice"),
        )
        .sort(GROUP)
    )
    counts.write_csv(output / "selection_counts.csv")
    overlap = (
        selection.select(
            "unit_id", "structure", "cohort", "window", "selection_half", "responsive"
        )
        .pivot(
            on="selection_half",
            index=["unit_id", "structure", "cohort", "window"],
            values="responsive",
        )
        .group_by("structure", "cohort", "window")
        .agg(
            pl.col("0").sum().alias("selected_primary"),
            pl.col("1").sum().alias("selected_reverse"),
            (pl.col("0") & pl.col("1")).sum().alias("selected_both"),
            (pl.col("0") | pl.col("1")).sum().alias("selected_either"),
        )
    )
    overlap.write_csv(output / "selection_overlap.csv")
    additional = []
    for key, frame in units.group_by(SESSION_GROUP, maintain_order=True):
        values = {
            "power_difference": frame["power_difference"].mean(),
            "power_difference_hz2": frame["power_difference_hz2"].mean(),
            "modality_selective_fraction": (frame["modality_q"] < 0.05).mean(),
            "target_selective_fraction": (frame["target_q"] < 0.05).mean(),
            "mean_abs_modality_hz": frame["modality_hz"].abs().mean(),
            "mean_abs_target_hz": frame["target_hz"].abs().mean(),
        }
        for metric, value in values.items():
            additional.append(
                {
                    **dict(zip(SESSION_GROUP, key)),
                    "metric": metric,
                    "estimate": value,
                    "n_selected": frame.height,
                    "null": [],
                }
            )
    sessions = pl.concat(
        [sessions, pl.from_dicts(additional)], how="vertical_relaxed"
    ).sort(*SESSION_GROUP, "metric")
    sessions.write_parquet(output / "session_summary.parquet")
    sessions.drop("null").write_csv(output / "session_summary.csv")
    results = []
    for key, frame in sessions.group_by([*GROUP, "metric"], maintain_order=True):
        frame = frame.sort("session_id")
        metric = key[-1]
        nulls = np.array(frame["null"].to_list()) if metric in POWER_METRICS else None
        stats = mouse_inference(
            frame["estimate"].to_numpy(),
            frame["subject_id"].to_numpy(),
            stable_seed(seed, *key),
            nulls,
        )
        if metric not in POWER_METRICS + DIFFERENCE_METRICS:
            stats["p_value"], stats["test"] = np.nan, "descriptive"
        results.append(
            {
                **dict(zip([*GROUP, "metric"], key)),
                **stats,
                "n_selected": frame["n_selected"].sum(),
            }
        )
    summary = pl.from_dicts(results)
    # Primary orientation and reversed orientation are separate, prespecified families.
    summary = adjust_by_group(summary, ["selection_half"], "p_value", "q_value").sort(
        *GROUP, "metric"
    )
    summary.write_csv(output / "area_summary.csv")
    mice = (
        sessions.group_by(*GROUP, "metric", "subject_id")
        .agg(
            pl.col("estimate").mean(),
            pl.len().alias("n_sessions"),
            pl.col("n_selected").sum(),
        )
        .sort(*GROUP, "metric", "subject_id")
    )
    mice.write_csv(output / "mouse_estimates.csv")
    return units, summary, mice, counts

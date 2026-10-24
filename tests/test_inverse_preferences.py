"""Conditional denominators, overlapping preferences, and background prevalence."""

import numpy as np
import polars as pl

from analyses.preference_context.inverse import (
    add_indicators,
    analyze,
    conditional_counts,
)


def sample():
    m = np.tile([0.01, 0.01, 0.01, 1.0], 6)
    t = np.tile([0.01, 1.0, 1.0, 0.01], 6)
    return pl.DataFrame(
        {
            "session_id": np.repeat(["s1", "s2", "s3"], 8),
            "subject_id": np.repeat(["m1", "m2", "m3"], 8),
            "structure": ["SCm"] * 24,
            "cohort": ["expert"] * 24,
            "window": ["early"] * 24,
            "selection_half": [0] * 24,
            "fit_subset": ["all"] * 24,
            "glm_converged": [True] * 24,
            "modality_q": m,
            "target_q": t,
            "dominant_class": np.where(
                m < 0.05, "modality_dominant", "target_dominant"
            ),
            "glm_baseline_p": np.tile([0.01] * 4 + [0.5] * 4, 3),
            "selection_baseline_hz": [10.0] * 24,
            "selection_response_rms_hz": [10.0] * 24,
        }
    )


def test_inverse_denominator_and_overlap():
    result = conditional_counts(add_indicators(sample()))
    row = result.filter(pl.col("baseline_context")).row(0, named=True)
    assert row["n_cells"] == 12
    assert row["n_modality"] == 9
    assert row["n_target"] == 6
    assert row["n_both"] == 3
    assert row["n_neither"] == 0
    assert row["n_modality"] + row["n_target"] > row["n_cells"]


def test_common_modality_preference_does_not_imply_baseline_enrichment():
    _, _, summary, _ = analyze(sample())
    common = summary.filter(
        pl.col("comparison") == "within_baseline", pl.col("fit_subset") == "all"
    ).row(0, named=True)
    assert common["left"] == 0.75 and common["right"] == 0.5
    assert common["estimate"] == 0.25
    assert common["n_mice"] == 3
    assert common["p_value"] == 0.25
    enrichment = summary.filter(
        pl.col("comparison") == "baseline_enrichment",
        pl.col("fit_subset") == "all",
        pl.col("adjustment") == "unmatched",
    )
    np.testing.assert_array_equal(enrichment["estimate"].to_numpy(), 0)

"""Saved-GLM linkage, QC-universe FDR, and animal-level inference."""

import numpy as np
import polars as pl
import pytest

from gain_glm.preference_context.glm.analysis import GLM_COLUMNS, compare, link_data


def inputs():
    n = 60
    ids = [f"unit{i}" for i in range(n)]
    sessions = [f"session{i // 20}" for i in range(n)]
    metadata = pl.DataFrame(
        {
            "unit_id": ids,
            "session_id": sessions,
            "subject_id": sessions,
            "structure": ["SCm"] * n,
            "cohort": ["expert"] * n,
        }
    )
    is_modality = np.arange(n) % 20 < 10
    preferences = metadata.with_columns(
        pl.lit("early").alias("window"),
        pl.lit(0).alias("selection_half"),
        pl.Series("modality_hz", np.where(is_modality, 4.0, 0.0)),
        pl.Series("target_hz", np.where(is_modality, 0.0, 4.0)),
        pl.Series("modality_q", np.where(is_modality, 0.01, 1.0)),
        pl.Series("target_q", np.where(is_modality, 1.0, 0.01)),
        pl.lit(10.0).alias("selection_response_rms_hz"),
    )
    covariates = preferences.select(
        "unit_id", "session_id", "window", "selection_half"
    ).with_columns(
        pl.lit(0).alias("preference_quarter"),
        pl.lit(10.0).alias("selection_baseline_hz"),
    )
    data = {"unit_id": ids, "session_id": sessions}
    for source, name in GLM_COLUMNS.items():
        if "converged" in name:
            data[source] = [True] * n
        elif name == "glm_baseline_p":
            data[source] = np.where(is_modality, 0.01, 0.8)
        elif name == "glm_n_shifts":
            data[source] = [99] * n
        else:
            data[source] = np.where(is_modality, 0.1, 0.01)
    return metadata, preferences, covariates, pl.DataFrame(data)


def test_join_excludes_missing_fits_and_fdr_uses_qc_universe():
    metadata, preferences, covariates, glm = inputs()
    # Keep only three selected cells, while FDR must still use all 60 QC cells.
    linked, coverage, universe = link_data(
        metadata, preferences.head(3), covariates, glm
    )
    np.testing.assert_allclose(linked["glm_baseline_q"], 0.02)
    assert universe.height == 60
    assert coverage["n_glm"].sum() == 60
    missing = glm.slice(1)
    linked, coverage, _ = link_data(metadata, preferences, covariates, missing)
    assert linked.height == 59
    assert coverage["n_glm"].sum() == 59
    assert "unit0" not in linked["unit_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        link_data(metadata, preferences, covariates, pl.concat([glm, glm.head(1)]))


def test_glm_comparison_preserves_mouse_unit_and_overlap_classes(tmp_path):
    linked, _, _ = link_data(*inputs())
    summary = compare(linked, tmp_path, 5)
    primary = summary.filter(
        pl.col("fit_subset") == "all",
        pl.col("comparison") == "dominant",
        pl.col("adjustment") == "unmatched",
        pl.col("metric") == "glm_nominal_probability",
    ).row(0, named=True)
    assert primary["estimate"] == 1
    assert primary["n_mice"] == 3
    assert primary["p_value"] == 0.25
    assert primary["n_left"] == 30
    assert primary["n_right"] == 30
    assert pl.read_csv(tmp_path / "matched_pairs.csv")["distance"].max() == 0

"""Scientific invariants and synthetic recovery tests for preference analysis."""

import numpy as np
import polars as pl

from gain_glm.context_preferences.analysis import aggregate, analyze_session
from gain_glm.context_preferences.data import select_areas
from gain_glm.context_preferences.statistics import (
    association_test,
    contrast_weights,
    estimate_preferences,
    fdr,
    modality_null,
    mouse_summary,
    permutation_p,
    spatial_test,
    split_trials,
)


def design(repeats=12):
    block = np.repeat(np.arange(6), 4 * repeats)
    stimulus = np.tile(np.repeat(np.arange(4), repeats), 6)
    context = block % 2
    half = split_trials(stimulus, block, 13)
    time = np.linspace(-1, 1, len(block))
    nuisance = np.column_stack([np.ones(len(block)), time, np.eye(4)[stimulus, 1:]])
    return stimulus, context, block, half, nuisance


def test_requested_areas_and_scm_pooling_preserve_source_labels():
    regions = [
        "SCiw",
        "SCig",
        "SCdg",
        "SCdw",
        "SCsg",
        "MRN",
        "ACAd",
        "ACAv",
        "ORBvl",
        "VISp",
        "VISp",
        None,
    ]
    units = pl.DataFrame(
        {
            "unit_id": [str(i) for i in range(len(regions))],
            "structure": regions,
            "session_id": ["s"] * len(regions),
            "subject_id": ["m"] * len(regions),
            "cohort": ["expert"] * len(regions),
        }
    )
    _, selected, counts = select_areas(units, minimum_area=2)
    assert set(selected["structure"]) == {"SCm", "MRN", "ACAd", "ACAv", "ORBvl", "VISp"}
    scm = selected.filter(pl.col("structure") == "SCm")
    assert set(scm["structure_original"]) == {"SCiw", "SCig", "SCdg"}
    assert scm.height == 3
    assert selected["unit_id"].n_unique() == selected.height
    assert (
        counts.filter(pl.col("structure") == "MRN")["selection_reason"][0]
        == "requested_inclusion"
    )
    _, threshold_only, _ = select_areas(units, minimum_area=3, included_areas=[])
    assert set(threshold_only["structure"]) == {"SCm"}


def test_balanced_contrasts_recover_known_effects_and_drift():
    s, c, b, h, z = design()
    baseline = (10 + 4 * c + 2 * z[:, 1])[None]
    response = (3 * (s < 2) + 2 * (s % 2 == 0))[None].astype(float)
    result, _, _, _, _ = estimate_preferences(baseline, response, s, c, b, h, z)
    np.testing.assert_allclose(result["modality"], 3)
    np.testing.assert_allclose(result["target"], 2)
    np.testing.assert_allclose(result["baseline"], 4)
    np.testing.assert_allclose(result["modality_target"], result["modality_nontarget"])


def test_single_context_keeps_sensory_tuning_but_not_context_contrast():
    s, c, b, h, z = design()
    c = np.ones_like(c)
    baseline = np.ones((1, len(s)))
    response = (s < 2)[None].astype(float)
    result, *_ = estimate_preferences(baseline, response, s, c, b, h, z)
    np.testing.assert_allclose(result["modality"], 1)
    assert np.isnan(result["baseline"]).all()
    assert np.isnan(result["baseline_raw"]).all()


def test_modality_shuffle_preserves_target_and_block_effects():
    s, c, b, h, _ = design()
    weights = contrast_weights(s, c, h)
    evoked = (5 * (s % 2 == 0) + 3 * b)[None].astype(float)
    null = modality_null(evoked, s, b, h, weights, 30, 8)
    # Shuffling within block/target/half cannot scramble these nuisance effects.
    np.testing.assert_allclose(null, 0, atol=1e-6)


def test_modality_label_null_detects_generalizing_tuning():
    s, c, b, h, z = design(20)
    rng = np.random.default_rng(12)
    slopes = np.linspace(-8, 8, 80)
    response = slopes[:, None] * (s < 2)[None] + rng.normal(size=(80, len(s)))
    result, _, _, scale, weights = estimate_preferences(
        np.ones_like(response), response, s, c, b, h, z
    )
    null = modality_null(response, s, b, h, weights, 99, 8)
    observed = np.mean(
        result["modality_target"]
        * result["modality_nontarget"][:, ::-1]
        / scale[:, None] ** 2
    )
    shuffled = np.mean(
        null[:, :, :, 1] * null[:, :, ::-1, 2] / scale[None, :, None] ** 2, axis=(1, 2)
    )
    assert permutation_p(observed, shuffled) == 0.01


def test_cross_half_relationship_avoids_shared_measurement_error():
    rng = np.random.default_rng(8)
    measurement_error = rng.normal(size=(2000, 2))
    baseline = measurement_error
    stimulus = -measurement_error + rng.normal(size=(2000, 2)) * 0.1
    # Same-half baseline subtraction would create a near-perfect anticorrelation.
    assert np.corrcoef(baseline[:, 0], stimulus[:, 0])[0, 1] < -0.99
    value, _ = association_test(baseline, stimulus, 49, 3)
    assert abs(value) < 0.08


def test_spatial_gradient_and_probe_confounds():
    rng = np.random.default_rng(7)
    coords = np.column_stack([np.arange(80) * 10, np.zeros(80), np.zeros(80)])
    x = np.linspace(-2, 2, 80)
    values = x[:, None] + rng.normal(0, 0.1, (80, 2))
    probes = np.zeros(80)
    value, null, meta = spatial_test(values, coords, probes, 99, 5)
    assert value > 0.7
    assert permutation_p(value, null) == 0.01
    assert meta["n_sites"] == 80
    probe = np.repeat([0, 1], 40)
    only_probe = np.tile(probe[:, None], (1, 2)).astype(float)
    value, _, _ = spatial_test(only_probe, coords, probe, 19, 5, within_probe=True)
    assert np.isnan(value)


def test_mouse_weighting_and_single_mouse_uncertainty():
    result = mouse_summary([0, 0, 0, 1], ["a", "a", "a", "b"], np.zeros((4, 20)), 1)
    assert result["estimate"] == 0.5
    assert result["n_sessions"] == 4
    one = mouse_summary([0, 1], ["a", "a"], np.zeros((2, 20)), 1)
    assert np.isnan(one["ci_low"])
    assert not one["population_inference"]


def test_fdr_retains_missing_values():
    q = fdr([0.01, 0.02, np.nan, 0.9])
    np.testing.assert_allclose(q[[0, 1, 3]], [0.03, 0.03, 0.9])
    assert np.isnan(q[2])


def test_session_to_report_tables(tmp_path):
    s, c, b, h, z = design()
    rng = np.random.default_rng(0)
    n = 25
    baseline = 10 + rng.normal(size=(n, len(s))) + np.linspace(-2, 2, n)[:, None] * c
    evoked = np.linspace(-3, 3, n)[:, None] * (s < 2) + rng.normal(size=baseline.shape)
    units = pl.DataFrame(
        {
            "unit_id": [f"test-{i:03}" for i in range(n)],
            "session_id": ["session"] * n,
            "subject_id": ["mouse"] * n,
            "structure": ["VISp"] * n,
            "cohort": ["expert"] * n,
            "firing_rate": [10.0] * n,
            "electrode_group_name": ["A"] * n,
            "ccf_ap": np.arange(n) * 20.0,
            "ccf_dv": [100.0] * n,
            "ccf_ml": [100.0] * n,
        }
    )
    arrays = {
        "baseline": baseline,
        "early": baseline + evoked,
        "late": baseline + evoked,
        "stimulus": s,
        "context": c,
        "block": b,
        "half": h,
        "nuisance": z,
        "times": np.arange(len(s)),
    }
    unit_frame, session_frame = analyze_session(
        {"session_id": "session", "subject_id": "mouse", "cohort": "expert"},
        units,
        arrays,
        19,
        2,
    )
    area, _, _ = aggregate(session_frame, unit_frame, tmp_path, 2)
    assert area.height > 10
    assert area["n_mice"].unique().to_list() == [1]
    assert (tmp_path / "unit_preferences.parquet").exists()

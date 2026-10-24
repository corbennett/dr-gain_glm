"""Selection independence, matched contrasts, null behavior, and aggregation."""

import numpy as np
import polars as pl
from scipy.stats import ttest_1samp

from gain_glm.context_preferences.statistics import contrast_weights, split_trials
from gain_glm.responsive_preferences.analysis import (
    aggregate,
    evaluate_session,
    prepare_session,
    select_responsive,
)
from gain_glm.responsive_preferences.statistics import (
    evaluation_design,
    factor_null,
    mouse_inference,
    preference_estimates,
    response_selection,
)


def design(n=24, repeats=16):
    block = np.repeat(np.arange(6), 4 * repeats)
    stimulus = np.tile(np.repeat(np.arange(4), repeats), 6)
    context = block % 2
    half = split_trials(stimulus, block, 21)
    rng = np.random.default_rng(8)
    baseline = np.full((n, len(stimulus)), 10.0)
    evoked = (
        5
        + 4 * (stimulus < 2)
        + 1 * (stimulus % 2 == 0)
        + rng.normal(0, 1.0, (n, len(stimulus)))
    )
    arrays = {
        "stimulus": stimulus,
        "block": block,
        "context": context,
        "half": half,
        "baseline": baseline,
        "early": baseline + evoked,
        "late": baseline + evoked,
        "unit_ids": np.array([f"u{i:03}" for i in range(n)]),
    }
    units = pl.DataFrame(
        {
            "unit_id": arrays["unit_ids"],
            "session_id": ["session"] * n,
            "subject_id": ["mouse"] * n,
            "cohort": ["expert"] * n,
            "structure": ["SCm"] * n,
        }
    )
    return units, arrays


def test_response_test_matches_paired_mean_test_and_corrects_conditions():
    _, a = design(4)
    y = a["early"] - a["baseline"]
    mask = a["half"] == 0
    result = response_selection(y, a["stimulus"], a["context"], mask)
    expected = []
    for c in (0, 1):
        for s in range(4):
            idx = mask & (a["context"] == c) & (a["stimulus"] == s)
            expected.append(ttest_1samp(y[:, idx], 0, axis=1).pvalue)
    expected = np.array(expected).T
    np.testing.assert_allclose(result["response_condition_p"], expected)
    np.testing.assert_allclose(
        result["response_p"], np.minimum(1, 8 * expected.min(axis=1))
    )


def test_selection_includes_common_response_and_suppression_but_not_silence():
    _, a = design(3)
    y = np.stack(
        [
            np.ones(len(a["half"])) * 5,
            np.ones(len(a["half"])) * -5,
            np.zeros(len(a["half"])),
        ]
    )
    result = response_selection(y, a["stimulus"], a["context"], a["half"] == 0)
    np.testing.assert_array_equal(result["response_p"], [0, 0, 1])
    np.testing.assert_allclose(result["normalization_hz"], [5, 5, 1])


def test_evaluation_perturbation_cannot_change_primary_selection():
    units, a = design(6)
    before = prepare_session(units, a, 3).filter(pl.col("selection_half") == 0)
    altered = {k: v.copy() for k, v in a.items()}
    idx = (a["half"] == 1) & (a["stimulus"] < 2)
    altered["early"][:, idx] += 100
    after = prepare_session(units, altered, 3).filter(pl.col("selection_half") == 0)
    np.testing.assert_array_equal(before["response_p"], after["response_p"])
    np.testing.assert_array_equal(before["normalization_hz"], after["normalization_hz"])
    assert (
        after.filter(pl.col("window") == "early")["modality_hz"].mean()
        > before["modality_hz"].mean() + 90
    )


def test_matched_contrasts_recover_relative_strength_and_reversal():
    _, a = design(1)
    keep, _, _, _, weights = evaluation_design(a, 0, 1, "session")
    s = a["stimulus"][keep]
    y = np.stack(
        [10 + 6 * (s < 2) + 2 * (s % 2 == 0), 10 + 2 * (s < 2) + 6 * (s % 2 == 0)]
    )
    result = preference_estimates(y, weights, np.array([2, 2]))
    np.testing.assert_allclose(result["power_difference"], [8, -8])
    np.testing.assert_allclose(result["power_difference_hz2"], [32, -32])


def test_both_factor_nulls_preserve_block_and_other_factor():
    _, a = design(1)
    _keep, s, b, h, w = evaluation_design(a, 0, 1, "session")
    for factor, signal in [("modality", 4 * (s % 2 == 0)), ("target", 4 * (s < 2))]:
        y = (signal + 2 * b)[None].astype(float)
        null = factor_null(y, s, b, h, w, factor, 31, 1)
        np.testing.assert_allclose(null, 0, atol=1e-6)


def test_nonresponsive_poisson_selection_is_not_anticonservative():
    _, a = design(1)
    rng = np.random.default_rng(72)
    # Equal firing rates, unequal pre/post durations, several sparse-spiking regimes.
    rates = np.repeat([0.5, 2, 10, 40], 500)
    baseline = (
        rng.poisson(rates[:, None] * 0.2, size=(len(rates), len(a["half"]))) / 0.2
    )
    evoked = rng.poisson(rates[:, None] * 0.1, size=baseline.shape) / 0.1 - baseline
    result = response_selection(evoked, a["stimulus"], a["context"], a["half"] == 0)
    assert np.mean(result["response_p"] < 0.05) < 0.06


def test_crossvalidated_powers_do_not_inflate_noise_into_preference():
    _, a = design(1, repeats=12)
    rng = np.random.default_rng(4)
    weights = contrast_weights(a["stimulus"], a["context"], a["half"])
    y = 10 + rng.normal(0, 8, (4000, len(a["half"])))
    result = preference_estimates(y, weights, np.ones(len(y)))
    standard_error = result["powers"].std(axis=0, ddof=1) / np.sqrt(len(y))
    assert np.all(np.abs(result["powers"].mean(axis=0)) < 4 * standard_error)
    assert np.mean(result["contrasts"].mean(axis=1) ** 2) > 0.5


def test_mouse_inference_weights_mice_equally_and_flags_single_mouse():
    result = mouse_inference([0, 0, 0, 1], ["a", "a", "a", "b"], 2)
    assert result["estimate"] == 0.5
    one = mouse_inference([1, 2], ["a", "a"], 2)
    assert np.isnan(one["p_value"]) and np.isnan(one["ci_low"])
    assert not one["population_inference"]


def test_end_to_end_same_selected_cells_both_factors(tmp_path):
    units, arrays = design()
    selection = select_responsive(prepare_session(units, arrays, 2), 0.05)
    assert selection["responsive"].all()
    uf, sf = evaluate_session(selection, units, arrays, 2, 39)
    result, summary, _mice, counts = aggregate(selection, uf, sf, tmp_path, 2)
    assert result.height == units.height * 4
    assert (
        result["modality_p"].is_not_null().all()
        and result["target_p"].is_not_null().all()
    )
    assert summary.filter(pl.col("metric") == "power_difference")["estimate"].min() > 0
    assert (
        summary.filter(pl.col("metric") == "power_difference")["p_value"].is_nan().all()
    )
    assert counts["n_selected"].sum() == units.height * 4
    assert (tmp_path / "selection_overlap.csv").exists()

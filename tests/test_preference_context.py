"""Independent data use, block-level calibration, overlap, and paired comparisons."""

import numpy as np
import polars as pl
from scipy.special import stdtr

from gain_glm.context_preferences.statistics import split_trials
from gain_glm.preference_context.analysis import aggregate, evaluate_session
from gain_glm.preference_context.statistics import (
    block_pair_power,
    cluster_context_test,
    match_groups,
    preference_categories,
)
from gain_glm.responsive_preferences.statistics import evaluation_design


def design(n=60, sid="session", seed=1):
    rng = np.random.default_rng(seed)
    block = np.repeat(np.arange(6), 64)
    stimulus = np.tile(np.repeat(np.arange(4), 16), 6)
    context = block % 2
    time = np.linspace(-1, 1, len(block))
    nuisance = np.column_stack([np.ones(len(block)), time])
    mod = np.arange(n) < n // 2
    base = 20 + 8 * mod[:, None] * context + rng.normal(0, 1, (n, len(block)))
    evoked = (
        3
        + 12 * mod[:, None] * (stimulus < 2)
        + 12 * (~mod[:, None]) * (stimulus % 2 == 0)
    )
    evoked = evoked + rng.normal(0, 1, base.shape)
    a = {
        "stimulus": stimulus,
        "context": context,
        "block": block,
        "nuisance": nuisance,
        "half": split_trials(stimulus, block, 21),
        "baseline": base,
        "early": base + evoked,
        "late": base + evoked,
        "unit_ids": np.array([f"{sid}_{i:03d}" for i in range(n)]),
    }
    u = pl.DataFrame(
        {
            "unit_id": a["unit_ids"],
            "session_id": [sid] * n,
            "subject_id": [sid] * n,
            "structure": ["SCm"] * n,
            "cohort": ["expert"] * n,
        }
    )
    selections = [
        u.with_columns(
            pl.lit(window).alias("window"),
            pl.lit(half).alias("selection_half"),
            pl.lit(0.001).alias("response_q"),
            pl.lit(10.0).alias("selection_response_rms_hz"),
        )
        for window in ("early", "late")
        for half in (0, 1)
    ]
    return u, a, pl.concat(selections)


def test_cr2_matches_group_mean_variance_and_four_degrees_of_freedom():
    rng = np.random.default_rng(4)
    block = np.repeat(np.arange(6), 20)
    context = block % 2
    y = rng.normal(size=(100, 6))[:, block] + rng.normal(0, 0.5, (100, 120))
    result = cluster_context_test(y, context, np.ones((120, 1)), block, cubic=False)
    means = np.stack([y[:, block == b].mean(axis=1) for b in range(6)], axis=1)
    expected = means[:, 1::2].mean(axis=1) - means[:, ::2].mean(axis=1)
    se = np.sqrt(
        means[:, 1::2].var(axis=1, ddof=1) / 3 + means[:, ::2].var(axis=1, ddof=1) / 3
    )
    np.testing.assert_allclose(result["hz"], expected, atol=1e-12)
    np.testing.assert_allclose(result["se_hz"], se, atol=1e-12)
    np.testing.assert_allclose(result["df"], 4)
    np.testing.assert_allclose(
        result["p"], 2 * stdtr(4, -np.abs(expected / se)), atol=1e-12
    )


def test_block_shock_null_is_calibrated_and_silence_not_significant():
    rng = np.random.default_rng(84)
    _, a, _ = design()
    # Strong serial dependence: random block-wide offsets dwarf trial noise.
    y = rng.normal(0, 6, (5000, 6))[:, a["block"]] + rng.normal(
        size=(5000, len(a["block"]))
    )
    result = cluster_context_test(y, a["context"], a["nuisance"], a["block"])
    assert 0.02 < np.mean(result["p"] < 0.05) < 0.075
    silent = cluster_context_test(
        np.zeros((1, len(a["block"]))), a["context"], a["nuisance"], a["block"]
    )
    assert silent["p"][0] == 1


def test_pair_power_rejects_linear_drift_and_recovers_stable_context():
    _, a, _ = design()
    c, z, b = a["context"], a["nuisance"], a["block"]
    y = np.stack([20 + 10 * z[:, 1], 20 + 10 * z[:, 1] + 4 * c])
    power, coeff = block_pair_power(y, c, z, b, np.array([2.0, 2.0]))
    np.testing.assert_allclose(power, [0, 4], atol=1e-12)
    np.testing.assert_allclose(coeff[1], [4, 4, 4], atol=1e-12)


def test_overlap_and_dominance_are_explicit():
    exclusive, dominant = preference_categories(
        np.array([0.01, 0.9, 0.01, 0.9, 0.01]),
        np.array([0.9, 0.01, 0.01, 0.9, 0.01]),
        np.array([4, 1, 4, 4, 1]),
        np.array([1, -4, -3, 1, 2]),
    )
    assert exclusive.tolist() == [
        "modality_only",
        "target_only",
        "both",
        "neither",
        "both",
    ]
    assert dominant.tolist() == [
        "modality_dominant",
        "target_dominant",
        "modality_dominant",
        "unclassified",
        "target_dominant",
    ]


def test_baseline_outcomes_cannot_change_preference_or_selection():
    u, a, s = design(12)
    first = evaluate_session(s, u, a, 8, 49).filter(
        pl.col("selection_half") == 0, pl.col("preference_quarter") == 0
    )
    keep, _, _, split, _ = evaluation_design(a, 0, 8, "session")
    indices = np.flatnonzero(keep)[split == 1]
    changed = {k: v.copy() for k, v in a.items()}
    changed["baseline"][:, indices] += 30 * a["context"][indices]
    after = evaluate_session(s, u, changed, 8, 49).filter(
        pl.col("selection_half") == 0, pl.col("preference_quarter") == 0
    )
    for name in (
        "modality_hz",
        "target_hz",
        "modality_p",
        "target_p",
        "response_q",
        "selection_baseline_hz",
    ):
        np.testing.assert_array_equal(first[name], after[name])
    np.testing.assert_allclose(
        after["baseline_hz"].to_numpy() - first["baseline_hz"].to_numpy(),
        30,
        atol=1e-10,
    )


def test_matching_does_not_reuse_cells_or_accept_out_of_caliper_pairs():
    features = np.array([[0.0, 1], [1.0, 1], [2.0, 1], [0.01, 1], [1.01, 1], [50, 1]])
    pairs, distances, _ = match_groups(
        features, np.array([0, 1, 2]), np.array([3, 4, 5]), 0.2
    )
    assert pairs.shape == (2, 2)
    assert len(np.unique(pairs)) == 4
    assert np.all(distances <= 0.2)


def test_end_to_end_recovers_association_and_preserves_mouse_unit(tmp_path):
    frames = []
    for i in range(3):
        u, a, s = design(sid=f"mouse{i}", seed=8 + i)
        frames.append(evaluate_session(s, u, a, 12, 99))
    units, summary, mice, _ = aggregate(pl.concat(frames), tmp_path, 11)
    row = summary.filter(
        pl.col("window") == "early",
        pl.col("selection_half") == 0,
        pl.col("preference_quarter") == 0,
        pl.col("comparison") == "dominant",
        pl.col("adjustment") == "unmatched",
        pl.col("metric") == "encoding_probability",
    ).row(0, named=True)
    assert row["estimate"] > 0.7
    assert row["n_mice"] == 3
    assert row["n_sessions"] == 3
    assert row["p_value"] >= 0.25  # three mice cannot give a smaller two-sided sign p
    assert units.height == 3 * 60 * 8
    assert mice["subject_id"].n_unique() == 3
    # No detections must not be presented as a precisely estimated equality.
    empty_dir = tmp_path / "no_detections"
    empty_dir.mkdir()
    no_detection = pl.concat(frames).with_columns(
        pl.lit(1.0).alias("baseline_p"), pl.lit(1.0).alias("baseline_linear_p")
    )
    _, empty_summary, _, _ = aggregate(no_detection, empty_dir, 11)
    zero_rows = empty_summary.filter(
        pl.col("metric").str.ends_with("encoding_probability")
    )
    assert set(zero_rows["test"]) == {"no_outcome_variation"}
    assert np.isnan(zero_rows["ci_low"].to_numpy()).all()
    assert np.isnan(zero_rows["p_value"].to_numpy()).all()

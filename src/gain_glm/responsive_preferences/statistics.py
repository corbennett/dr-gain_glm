"""Selection, held-out contrasts, and matched tests for the two factors."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.special import stdtr

from gain_glm.context_preferences.statistics import (
    contrast_weights,
    fdr,
    permutation_p,
    split_trials,
    stable_seed,
)

FACTOR_WEIGHTS = np.array([[0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5]])
FACTORS = ("modality", "target")


def response_selection(evoked, stimulus, context, mask, minimum=5):
    """Two-sided paired-rate t tests, with a Bonferroni any-condition test.

    evoked is post-stimulus rate minus same-trial baseline rate. Each available
    context × stimulus is tested separately, retaining excitation, suppression,
    and responses that reverse sign between contexts. Only selection trials
    enter the test or the common response-strength denominator.
    """
    means, ps = [], []
    for c in np.unique(context):
        for s in range(4):
            idx = mask & (context == c) & (stimulus == s)
            n = int(idx.sum())
            if n < minimum:
                raise ValueError(f"Selection support: context={c}, stimulus={s}, n={n}")
            values = np.asarray(evoked[:, idx], dtype=float)
            mean = values.mean(axis=1)
            se = values.std(axis=1, ddof=1) / np.sqrt(n)
            statistic = np.divide(mean, se, out=np.zeros_like(mean), where=se > 0)
            p = 2 * stdtr(n - 1, -np.abs(statistic))
            p[(se == 0) & (mean != 0)] = 0
            p[(se == 0) & (mean == 0)] = 1
            means.append(mean)
            ps.append(p)
    means, ps = np.stack(means, axis=1), np.stack(ps, axis=1)
    omnibus = np.minimum(1, ps.min(axis=1) * ps.shape[1])
    rms = np.sqrt(np.mean(means**2, axis=1))
    return {
        "response_p": omnibus,
        "selection_response_rms_hz": rms,
        "normalization_hz": np.maximum(rms, 1.0),
        "response_condition_means": means,
        "response_condition_p": ps,
    }


def evaluation_design(arrays, selection_half, seed, session_id):
    keep = arrays["half"] != selection_half
    stimulus, context, block = (
        arrays[k][keep] for k in ("stimulus", "context", "block")
    )
    split = split_trials(
        stimulus, block, stable_seed(seed, session_id, selection_half, "evaluation")
    )
    weights = contrast_weights(stimulus, context, split, minimum=2)
    return keep, stimulus, block, split, weights


def preference_estimates(evoked, weights, normalization):
    means = np.einsum("ut,hst->uhs", evoked, weights)
    contrasts = means @ FACTOR_WEIGHTS.T  # neuron × evaluation split × factor
    scaled = contrasts / normalization[:, None, None]
    powers = scaled[:, 0] * scaled[:, 1]
    return {
        "means": means,
        "contrasts": contrasts,
        "scaled": scaled,
        "powers": powers,
        "power_difference": powers[:, 0] - powers[:, 1],
        "power_difference_hz2": contrasts[:, 0, 0] * contrasts[:, 1, 0]
        - contrasts[:, 0, 1] * contrasts[:, 1, 1],
    }


def factor_null(evoked, stimulus, block, split, weights, factor, permutations, seed):
    """Matched trial-label nulls with fixed selection and simultaneous-cell shuffles.

    Modality swaps are within target status; target swaps are within modality.
    Both preserve actual block and evaluation split. The two factors have equal
    contrast coefficient norms and use exactly the same evaluation trials.
    """
    factor_id = FACTORS.index(factor)
    nuisance = stimulus % 2 if factor == "modality" else stimulus // 2
    w = np.einsum("s,hst->ht", FACTOR_WEIGHTS[factor_id], weights)
    groups = [
        np.flatnonzero((block == b) & (nuisance == n) & (split == h))
        for b in np.unique(block)
        for n in (0, 1)
        for h in (0, 1)
    ]
    rng = np.random.default_rng(seed)
    result = np.empty((permutations, len(evoked), 2), dtype=np.float32)
    for start in range(0, permutations, 64):
        stop = min(start + 64, permutations)
        shuffled = np.empty((stop - start, 2, len(stimulus)))
        for j in range(stop - start):
            order = np.arange(len(stimulus))
            for idx in groups:
                order[idx] = rng.permutation(idx)
            shuffled[j] = w[:, order]
        result[start:stop] = (
            (evoked @ shuffled.reshape(-1, len(stimulus)).T)
            .reshape(len(evoked), stop - start, 2)
            .transpose(1, 0, 2)
        )
    return result


def unit_pvalues(observed, null):
    """Two-sided magnitude test with a tolerance for floating-point ties."""
    observed = np.abs(observed)
    tol = 1e-7 * np.maximum(1, observed)
    return (1 + np.sum(np.abs(null) >= observed[None] - tol[None], axis=0)) / (
        len(null) + 1
    )


def mouse_inference(values, subjects, seed, nulls=None, bootstraps=10000):
    """Equal-session, equal-mouse mean; CI and optional paired mouse sign test.

    With nulls, p is the conditional trial-shuffle p. Without nulls, p tests a
    zero mean paired difference by sign-flipping whole mouse effects; this
    assumes symmetry under the null. Fewer than three mice get no population p.
    """
    values, subjects = np.asarray(values, float), np.asarray(subjects)
    unique = np.unique(subjects)
    means = np.array([values[subjects == m].mean() for m in unique])
    rng = np.random.default_rng(seed)
    estimate = means.mean()
    ci = [np.nan, np.nan]
    if len(means) >= 2:
        samples = means[rng.integers(len(means), size=(bootstraps, len(means)))].mean(
            axis=1
        )
        ci = np.quantile(samples, [0.025, 0.975])
    null_mean = np.nan
    p = np.nan
    if nulls is not None:
        nulls = np.asarray(nulls)
        null = np.array([nulls[subjects == m].mean(axis=0) for m in unique]).mean(
            axis=0
        )
        null_mean = null.mean()
        p = permutation_p(estimate, null)
        test = "conditional_trial_shuffle"
    else:
        test = "paired_mouse_sign_flip"
        if len(means) >= 3:
            if len(means) <= 15:
                signs = np.array(list(itertools.product((-1, 1), repeat=len(means))))
                null = (signs * means).mean(axis=1)
                p = np.mean(np.abs(null) >= abs(estimate) - 1e-12)
            else:
                signs = rng.choice([-1, 1], size=(19999, len(means)))
                null = (signs * means).mean(axis=1)
                p = (1 + np.sum(np.abs(null) >= abs(estimate) - 1e-12)) / (
                    len(null) + 1
                )
    return {
        "estimate": estimate,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "p_value": p,
        "null_mean": null_mean,
        "test": test,
        "n_mice": len(means),
        "n_sessions": len(values),
        "population_inference": len(means) >= 3,
    }


def adjust_by_group(frame, keys, p_column, q_column):
    import polars as pl

    return pl.concat(
        [
            f.with_columns(pl.Series(q_column, fdr(f[p_column].to_numpy())))
            for _, f in frame.group_by(keys, maintain_order=True)
        ]
    )

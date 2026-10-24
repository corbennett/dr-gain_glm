"""Numerical methods, independent of NWB and remote data access.

Contrasts are differences in baseline-subtracted Hz, balanced over stimulus
identities and available contexts. Cross-fitted quantities always pair estimates
from disjoint trial halves. Neurons are never treated as independent animals.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

CONTRASTS = {
    # Order: visual target, visual nontarget, auditory target, auditory nontarget.
    "modality": np.array([0.5, 0.5, -0.5, -0.5]),
    "target": np.array([0.5, -0.5, 0.5, -0.5]),
    "target_visual": np.array([1.0, -1.0, 0.0, 0.0]),
    "target_auditory": np.array([0.0, 0.0, 1.0, -1.0]),
    "modality_target": np.array([1.0, 0.0, -1.0, 0.0]),
    "modality_nontarget": np.array([0.0, 1.0, 0.0, -1.0]),
}


def stable_seed(seed: int, *parts: object) -> int:
    text = "|".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def split_trials(stimulus, block, seed):
    """Balance random halves within each actual block × stimulus identity."""
    rng = np.random.default_rng(seed)
    half = np.full(len(stimulus), -1, dtype=int)
    for b in np.unique(block):
        for s in np.unique(stimulus):
            idx = np.flatnonzero((block == b) & (stimulus == s))
            idx = rng.permutation(idx)
            # Randomize which half receives an odd leftover trial.
            half[idx] = (np.arange(len(idx)) + rng.integers(2)) % 2
    return half


def contrast_weights(stimulus, context, half, minimum=5):
    """Trial weights for each identity, with contexts weighted equally.

    Single-context sessions retain sensory contrasts, but cannot provide a
    context contrast. Require every stimulus in every available context/half.
    """
    weights = np.zeros((2, 4, len(stimulus)))
    contexts = np.unique(context)
    for h in range(2):
        for s in range(4):
            for c in contexts:
                idx = np.flatnonzero((stimulus == s) & (context == c) & (half == h))
                if len(idx) < minimum:
                    raise ValueError(
                        f"Insufficient trials: half={h}, stimulus={s}, context={c}, "
                        f"n={len(idx)} < {minimum}"
                    )
                weights[h, s, idx] = 1 / (len(contexts) * len(idx))
    return weights


def residual_contrast(context, nuisance, mask):
    """OLS coefficient weights for visual context, controlling nuisance terms."""
    w = np.zeros(len(context))
    if len(np.unique(context[mask])) < 2:
        return np.full(len(context), np.nan)
    c = context[mask].astype(float)
    z = nuisance[mask]
    residual = c - z @ np.linalg.lstsq(z, c, rcond=None)[0]
    denom = residual @ c
    if denom < 1e-8:
        return np.full(len(context), np.nan)
    w[mask] = residual / denom
    return w


def estimate_preferences(
    baseline, evoked, stimulus, context, block, half, nuisance, minimum=5
):
    """Return unit × half estimates and unit × half × stimulus evoked means."""
    weights = contrast_weights(stimulus, context, half, minimum)
    means = np.einsum("ut,hst->uhs", evoked, weights)
    base_means = np.einsum("ut,hst->uhs", baseline, weights)
    estimates = {name: means @ w for name, w in CONTRASTS.items()}
    estimates["baseline"] = np.column_stack(
        [baseline @ residual_contrast(context, nuisance, half == h) for h in range(2)]
    )
    raw_weights = np.zeros((2, len(stimulus)))
    if len(np.unique(context)) == 2:
        for h in range(2):
            for s in range(4):
                for c in (0, 1):
                    idx = np.flatnonzero((half == h) & (stimulus == s) & (context == c))
                    raw_weights[h, idx] = (2 * c - 1) / (4 * len(idx))
        estimates["baseline_raw"] = baseline @ raw_weights.T
    else:
        estimates["baseline_raw"] = np.full((len(baseline), 2), np.nan)
    # Cross-validated response energy includes both excitation and suppression.
    estimates["visual_energy"] = np.mean(means[:, 0, :2] * means[:, 1, :2], axis=1)
    estimates["auditory_energy"] = np.mean(means[:, 0, 2:] * means[:, 1, 2:], axis=1)
    # Label-invariant scale: modulation-free residual SD within each block,
    # floored at 1 Hz. This prevents high-rate cells dominating population power.
    residuals = evoked.copy()
    for b in np.unique(block):
        idx = block == b
        residuals[:, idx] -= residuals[:, idx].mean(axis=1, keepdims=True)
    scale = np.maximum(np.std(residuals, axis=1, ddof=1), 1.0)
    return estimates, means, base_means, scale, weights


def modality_null(evoked, stimulus, block, half, weights, permutations, seed):
    """Shuffle V/A assignments within block × target-status × trial-half.

    A shuffle is shared by all simultaneously recorded cells, retaining their
    dependence. Target and nontarget strata are shuffled independently, testing
    generalization to the other stimulus identity as well as modality tuning.
    """
    rng = np.random.default_rng(seed)
    n = len(stimulus)
    w = np.einsum(
        "ks,hst->hkt",
        np.stack(
            [
                CONTRASTS["modality"],
                CONTRASTS["modality_target"],
                CONTRASTS["modality_nontarget"],
            ]
        ),
        weights,
    )
    groups = [
        np.flatnonzero((block == b) & (stimulus % 2 == t) & (half == h))
        for b in np.unique(block)
        for t in (0, 1)
        for h in (0, 1)
    ]
    result = np.empty((permutations, len(evoked), 2, 3), dtype=np.float32)
    for start in range(0, permutations, 32):
        end = min(start + 32, permutations)
        shuffled = np.empty((end - start, 2, 3, n))
        for j in range(end - start):
            order = np.arange(n)
            for idx in groups:
                order[idx] = rng.permutation(idx)
            shuffled[j] = w[:, :, order]
        flat = shuffled.reshape(-1, n)
        result[start:end] = (
            (evoked @ flat.T)
            .reshape(
                len(evoked),
                end - start,
                2,
                3,
            )
            .transpose(1, 0, 2, 3)
        )
    return result


def permutation_p(observed, null, *, two_sided=False):
    null = np.asarray(null)
    if not np.isfinite(observed) or not np.isfinite(null).all():
        return np.nan
    if two_sided:
        # Equal-tail test works for spatial nulls whose expectation is not zero.
        return min(
            1.0,
            2
            * min(
                (1 + np.sum(null >= observed)) / (len(null) + 1),
                (1 + np.sum(null <= observed)) / (len(null) + 1),
            ),
        )
    return (1 + np.sum(null >= observed)) / (len(null) + 1)


def fdr(p):
    """Benjamini-Hochberg, preserving missing hypotheses."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    order = idx[np.argsort(p[idx])]
    if len(order):
        adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
        q[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return q


def correlation(a, b):
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 4:
        return np.nan
    a, b = np.asarray(a)[keep], np.asarray(b)[keep]
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 1e-12 else np.nan


def cross_rank_correlation(a, b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan
    return 0.5 * (
        correlation(rankdata(a[:, 0]), rankdata(b[:, 1]))
        + correlation(rankdata(a[:, 1]), rankdata(b[:, 0]))
    )


def association_test(a, b, permutations, seed, strata=None):
    """Cross-half Spearman; shuffle paired halves together between neurons."""
    keep = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a, b = a[keep], b[keep]
    if len(a) < 10:
        return np.nan, np.full(permutations, np.nan)
    if strata is None:
        strata = np.zeros(len(a), dtype=int)
    else:
        strata = np.asarray(strata)[keep]
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    a = np.column_stack([rankdata(a[:, h]) for h in range(2)])
    b = np.column_stack([rankdata(b[:, h]) for h in range(2)])
    for x in (a, b):
        x -= x.mean(axis=0)
        norms = np.linalg.norm(x, axis=0)
        if np.any(norms < 1e-12):
            return np.nan, np.full(permutations, np.nan)
        x /= norms
    observed = float(np.sum(a * b[:, ::-1]) / 2)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for p in range(permutations):
        idx = np.arange(len(a))
        for g in groups:
            idx[g] = rng.permutation(g)
        null[p] = np.sum(a[idx] * b[:, ::-1]) / 2
    return observed, null


def neighbor_edges(coordinates, probes, *, within_probe, k=10, max_distance=500):
    """Symmetric union of k-nearest-neighbor edges, at most 500 µm apart.

    Coordinates are recording-site estimates; identical sites are legitimate
    neighbors, but self-edges are removed. Lexical input ordering makes ties
    reproducible. Unique recording-site counts are reported separately.
    """
    groups = np.unique(probes) if within_probe else [None]
    pairs = set()
    for group in groups:
        idx = (
            np.flatnonzero(probes == group) if within_probe else np.arange(len(probes))
        )
        if len(idx) < 2:
            continue
        distance, neighbors = cKDTree(coordinates[idx]).query(
            coordinates[idx],
            k=min(k + 1, len(idx)),
        )
        for i in range(len(idx)):
            for d, j in zip(np.atleast_1d(distance[i]), np.atleast_1d(neighbors[i])):
                if i != j and d <= max_distance:
                    pairs.add(tuple(sorted((int(idx[i]), int(idx[j])))))
    return np.asarray(sorted(pairs), dtype=int).reshape(-1, 2)


def spatial_test(
    values, coordinates, probes, permutations, seed, *, within_probe=False
):
    """Cross-half Moran-type neighbor covariance and paired-location null."""
    keep = np.isfinite(values).all(axis=1) & np.isfinite(coordinates).all(axis=1)
    values, coordinates, probes = values[keep], coordinates[keep], probes[keep]
    sites = len(np.unique(coordinates, axis=0))
    meta = {
        "n_cells": len(values),
        "n_sites": sites,
        "n_probes": len(np.unique(probes)),
    }
    null = np.full(permutations, np.nan)
    if len(values) < 20 or sites < 5:
        return np.nan, null, dict(meta, n_edges=0)
    edges = neighbor_edges(coordinates, probes, within_probe=within_probe)
    meta["n_edges"] = len(edges)
    if len(edges) < 10:
        return np.nan, null, meta
    x = values.copy()
    groups = (
        [np.flatnonzero(probes == p) for p in np.unique(probes)]
        if within_probe
        else [np.arange(len(x))]
    )
    for idx in groups:
        x[idx] -= x[idx].mean(axis=0)
    denom = np.sqrt(np.sum(x[:, 0] ** 2) * np.sum(x[:, 1] ** 2))
    if denom < 1e-12:
        return np.nan, null, meta
    i, j = edges.T
    factor = len(x) / (2 * len(edges) * denom)

    def statistic(y):
        return factor * np.sum(y[i, 0] * y[j, 1] + y[i, 1] * y[j, 0])

    observed = float(statistic(x))
    rng = np.random.default_rng(seed)
    for p in range(permutations):
        idx = np.arange(len(x))
        for g in groups:
            idx[g] = rng.permutation(g)
        null[p] = statistic(x[idx])
    return observed, null, meta


def mouse_summary(values, subjects, nulls, seed, bootstraps=2000):
    """Equal-session mean within mouse, then equal-mouse inference.

    Permutation nulls retain simultaneous-cell dependence within sessions.
    Bootstrap CIs resample mice, not individual neurons. A single mouse receives
    no population CI. Conditional permutation p-values remain explicitly labeled.
    """
    values, subjects, nulls = (
        np.asarray(values),
        np.asarray(subjects),
        np.asarray(nulls),
    )
    keep = np.isfinite(values) & np.isfinite(nulls).all(axis=1)
    values, subjects, nulls = values[keep], subjects[keep], nulls[keep]
    if not len(values):
        return None
    mouse_values, mouse_nulls = [], []
    for mouse in np.unique(subjects):
        mask = subjects == mouse
        mouse_values.append(values[mask].mean())
        mouse_nulls.append(nulls[mask].mean(axis=0))
    mouse_values, mouse_nulls = np.array(mouse_values), np.array(mouse_nulls)
    observed, null = mouse_values.mean(), mouse_nulls.mean(axis=0)
    centered = mouse_values - mouse_nulls.mean(axis=1)
    if len(mouse_values) >= 2:
        rng = np.random.default_rng(seed)
        idx = rng.integers(len(mouse_values), size=(bootstraps, len(mouse_values)))
        ci = np.quantile(mouse_values[idx].mean(axis=1), [0.025, 0.975])
        effect_ci = np.quantile(centered[idx].mean(axis=1), [0.025, 0.975])
    else:
        ci = effect_ci = np.array([np.nan, np.nan])
    return {
        "estimate": float(observed),
        "ci_low": ci[0],
        "ci_high": ci[1],
        "null_mean": float(null.mean()),
        "excess_over_null": float(observed - null.mean()),
        "excess_ci_low": effect_ci[0],
        "excess_ci_high": effect_ci[1],
        "p_upper": permutation_p(observed, null),
        "p_two_sided": permutation_p(observed, null, two_sided=True),
        "n_sessions": len(values),
        "n_mice": len(mouse_values),
        "population_inference": len(mouse_values) >= 3,
    }

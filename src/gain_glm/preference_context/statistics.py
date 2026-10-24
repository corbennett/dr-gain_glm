"""Block-aware baseline tests and within-session population comparisons."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import stdtr
from scipy.stats import rankdata


def cluster_context_test(baseline, context, nuisance, block, cubic=True):
    """OLS context coefficient, CR2 variance, and Satterthwaite t test.

    Random-intercept working covariance (within-block correlation 0.5);
    actual context blocks are the clusters. Uses
    the full hat matrix (no absorbed effects). The coefficient is visual minus
    auditory context. Rank-deficient nuisance columns are allowed, but the
    context coefficient must be estimable. No independent-trial t test is used.
    """
    y = np.asarray(baseline, float)
    z = np.asarray(nuisance, float)
    if cubic:
        z = np.column_stack([z, z[:, 1] ** 2, z[:, 1] ** 3])
    c = np.asarray(context, float)
    residual_c = c - z @ np.linalg.lstsq(z, c, rcond=None)[0]
    if residual_c @ residual_c < 1e-8:
        raise ValueError("Context coefficient is not estimable")
    x = np.column_stack([z, c])
    inverse = np.linalg.pinv(x)
    w = inverse[-1]
    beta = y @ w
    hat = x @ inverse
    residualizer = np.eye(len(c)) - hat
    target = 0.5 * np.eye(len(c)) + 0.5 * (block[:, None] == block)
    residual_target = residualizer @ target @ residualizer.T

    def symmetric_power(matrix, exponent):
        values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
        if values.min() < -1e-8:
            raise ValueError("Invalid cluster residual covariance")
        scale = np.zeros_like(values)
        good = values > 1e-10
        scale[good] = values[good] ** exponent
        return (vectors * scale) @ vectors.T

    # Each column maps raw outcomes to one adjusted cluster score.
    vectors = []
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        phi_root = symmetric_power(target[np.ix_(idx, idx)], 0.5)
        within = residual_target[np.ix_(idx, idx)]
        adjustment = (
            phi_root @ symmetric_power(phi_root @ within @ phi_root, -0.5) @ phi_root
        )
        vectors.append(residualizer[:, idx] @ adjustment @ w[idx])
    vectors = np.column_stack(vectors)
    variance = np.sum((y @ vectors) ** 2, axis=1)
    gram = vectors.T @ target @ vectors
    df = float(np.trace(gram) ** 2 / np.sum(gram**2))
    se = np.sqrt(np.maximum(variance, 0))
    statistic = np.divide(beta, se, out=np.zeros_like(beta), where=se > 1e-12)
    p = 2 * stdtr(df, -np.abs(statistic))
    p[(se <= 1e-12) & (np.abs(beta) <= 1e-10)] = 1
    p[(se <= 1e-12) & (np.abs(beta) > 1e-10)] = 0
    return {
        "hz": beta,
        "se_hz": se,
        "p": p,
        "df": df,
        "n_blocks": len(np.unique(block)),
        "rank": np.linalg.matrix_rank(x),
    }


def block_pair_power(baseline, context, nuisance, block, scale):
    """Products of context effects from disjoint adjacent block pairs.

    Each pair has its own nuisance coefficients and linear time trend. A
    product estimates squared stable context modulation without squaring
    trial noise, assuming errors in different pairs are independent.
    """
    blocks = np.unique(block)
    estimates = []
    for first in range(0, len(blocks) - 1, 2):
        idx = np.isin(block, blocks[first : first + 2])
        c, z = context[idx].astype(float), nuisance[idx]
        if len(np.unique(c)) != 2:
            continue
        residual = c - z @ np.linalg.lstsq(z, c, rcond=None)[0]
        denom = residual @ residual
        if denom <= 1e-8:
            continue
        estimates.append(baseline[:, idx] @ residual / denom)
    if len(estimates) < 2:
        return np.full(len(baseline), np.nan), np.full((len(baseline), 0), np.nan)
    estimates = np.column_stack(estimates)
    products = [
        estimates[:, i] * estimates[:, j]
        for i in range(estimates.shape[1])
        for j in range(i)
    ]
    return np.mean(products, axis=0) / scale**2, estimates


def selection_baseline_scale(baseline, block):
    residual = baseline.astype(float).copy()
    for b in np.unique(block):
        idx = block == b
        residual[:, idx] -= residual[:, idx].mean(axis=1, keepdims=True)
    return np.maximum(np.sqrt(np.mean(residual**2, axis=1)), 1)


def preference_categories(modality_q, target_q, modality_hz, target_hz, alpha=0.05):
    m, t = modality_q < alpha, target_q < alpha
    exclusive = np.select(
        [m & ~t, t & ~m, m & t], ["modality_only", "target_only", "both"], "neither"
    )
    dominant = np.full(len(m), "unclassified", dtype="U20")
    detected = m | t
    dominant[detected & (np.abs(modality_hz) > np.abs(target_hz))] = "modality_dominant"
    dominant[detected & (np.abs(target_hz) > np.abs(modality_hz))] = "target_dominant"
    return exclusive, dominant


def match_groups(features, left, right, caliper=1.5):
    """One-to-one minimum-distance matching, with a prespecified caliper.

    Standardize using both groups within this session/area. Optimize number of
    admissible matches before distance. Matching never sees baseline outcomes.
    """
    if not len(left) or not len(right):
        return np.empty((0, 2), int), np.empty(0), np.empty_like(features)
    subset = features[np.r_[left, right]]
    sd = subset.std(axis=0)
    sd[sd < 1e-8] = 1
    z = (features - subset.mean(axis=0)) / sd
    distances = np.linalg.norm(z[left, None] - z[right][None], axis=2)
    cost = np.where(distances <= caliper, distances, 1e6)
    i, j = linear_sum_assignment(cost)
    keep = distances[i, j] <= caliper
    pairs = np.column_stack([left[i[keep]], right[j[keep]]])
    return pairs, distances[i[keep], j[keep]], z


def partial_rank_correlation(balance, outcome, covariates):
    """Partial Spearman correlation, controlling the listed ranked covariates."""
    if len(balance) < 10 or np.ptp(balance) == 0 or np.ptp(outcome) == 0:
        return np.nan
    z = np.column_stack([np.ones(len(balance)), rankdata(covariates, axis=0)])
    y = rankdata(np.column_stack([balance, outcome]), axis=0)
    residual = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    if np.any(residual.std(axis=0) < 1e-8):
        return np.nan
    return float(np.corrcoef(residual.T)[0, 1])

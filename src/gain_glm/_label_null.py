"""Batched scoring of caller-supplied pseudo trial labels for permutation tests.

A permutation test freezes the reduced fit, re-inserts the tested terms under a
candidate trial label ``c``, and refits one linear (Ridge) subproblem. The
candidate label enters that subproblem only through trial-constant multipliers,
so per-trial sufficient statistics give every label's Ridge solution and
held-out squared error without forming one design matrix per label. The results
equal the per-label refits in :mod:`gain_glm.evaluation` up to floating-point
rounding.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._solver import _gain_design, _kernel_design, _kernel_drives
from .design import PreparedDesign
from .model import FitState

_LABEL_CHUNK = 128


def _trial_sums(
    trials: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique trials and ``values`` summed over rows of each trial."""
    unique, inverse = np.unique(trials, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(unique.size))
    return unique, np.add.reduceat(values[order], starts, axis=0)


class _GainStatistics:
    """Per-trial statistics of ``X(c) = [A | c_trial * D]`` and target ``r``."""

    def __init__(
        self,
        fixed: np.ndarray,
        tested: np.ndarray,
        target: np.ndarray,
        trials: np.ndarray,
    ):
        self.fixed_gram = fixed.T @ fixed
        self.fixed_target = fixed.T @ target
        self.target_energy = float(target @ target)
        self.trials, self.cross = _trial_sums(
            trials, fixed[:, :, None] * tested[:, None, :]
        )
        _, self.tested_gram = _trial_sums(
            trials, tested[:, :, None] * tested[:, None, :]
        )
        _, self.tested_target = _trial_sums(trials, tested * target[:, None])

    def normal_equations(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``labels`` has shape (n_labels, n_trials) indexed by ``self.trials``."""
        n_labels = labels.shape[0]
        n_fixed = self.fixed_gram.shape[0]
        n_tested = self.tested_gram.shape[1]
        cross = np.einsum("kt,tam->kam", labels, self.cross)
        gram = np.empty((n_labels, n_fixed + n_tested, n_fixed + n_tested))
        gram[:, :n_fixed, :n_fixed] = self.fixed_gram
        gram[:, :n_fixed, n_fixed:] = cross
        gram[:, n_fixed:, :n_fixed] = cross.transpose(0, 2, 1)
        gram[:, n_fixed:, n_fixed:] = np.einsum(
            "kt,tmn->kmn", labels**2, self.tested_gram
        )
        moment = np.empty((n_labels, n_fixed + n_tested))
        moment[:, :n_fixed] = self.fixed_target
        moment[:, n_fixed:] = labels @ self.tested_target
        return gram, moment


class _PredictorStatistics:
    """Per-trial statistics of ``X(c) = [1 | A | c_trial]`` and target ``y``."""

    def __init__(self, fixed: np.ndarray, target: np.ndarray, trials: np.ndarray):
        fixed = np.column_stack([np.ones(target.size), fixed])
        self.fixed_gram = fixed.T @ fixed
        self.fixed_target = fixed.T @ target
        self.target_energy = float(target @ target)
        self.trials, self.fixed_sums = _trial_sums(trials, fixed)
        _, self.target_sums = _trial_sums(trials, target)
        _, self.counts = _trial_sums(trials, np.ones(target.size))

    def normal_equations(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_labels = labels.shape[0]
        n_fixed = self.fixed_gram.shape[0]
        gram = np.empty((n_labels, n_fixed + 1, n_fixed + 1))
        gram[:, :n_fixed, :n_fixed] = self.fixed_gram
        cross = labels @ self.fixed_sums
        gram[:, :n_fixed, n_fixed] = cross
        gram[:, n_fixed, :n_fixed] = cross
        gram[:, n_fixed, n_fixed] = labels**2 @ self.counts
        moment = np.empty((n_labels, n_fixed + 1))
        moment[:, :n_fixed] = self.fixed_target
        moment[:, n_fixed] = labels @ self.target_sums
        return gram, moment


def _ridge_held_out_sse(
    train: _GainStatistics | _PredictorStatistics,
    test: _GainStatistics | _PredictorStatistics,
    trial_values: np.ndarray,
    alpha: float,
    penalty: np.ndarray,
) -> np.ndarray:
    """Solve one Ridge problem per label row on ``train`` and score it on ``test``."""
    sse = np.empty(trial_values.shape[0])
    ridge = alpha * np.diag(penalty)
    for start in range(0, trial_values.shape[0], _LABEL_CHUNK):
        chunk = trial_values[start : start + _LABEL_CHUNK]
        gram, moment = train.normal_equations(chunk[:, train.trials])
        coefficients = np.linalg.solve(gram + ridge, moment[..., None])[..., 0]
        test_gram, test_moment = test.normal_equations(chunk[:, test.trials])
        sse[start : start + chunk.shape[0]] = (
            test.target_energy
            - 2 * np.einsum("kg,kg->k", coefficients, test_moment)
            + np.einsum("kg,kgh,kh->k", coefficients, test_gram, coefficients)
        )
    return sse


def gain_label_sse(
    prepared: PreparedDesign,
    gain_terms: tuple[tuple[str, str], ...],
    trial_values: np.ndarray,
    state: FitState,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_blocks: Mapping[str, np.ndarray],
    test_blocks: Mapping[str, np.ndarray],
    train_gains: Mapping[str, np.ndarray],
    test_gains: Mapping[str, np.ndarray],
    train_trials: np.ndarray,
    test_trials: np.ndarray,
) -> np.ndarray:
    """Held-out SSE per label row when gains are refit with kernels frozen.

    Matches ``_fixed_kernel_shift_predictions``: the tested gain columns become
    ``label * drive`` and all gain coefficients are refit by Ridge without an
    intercept at the reduced fit's gain penalty.
    """
    alpha = state.iterations[-1].gain_alpha
    if alpha is None:
        raise RuntimeError(
            "gain permutation test requires a gain penalty from the reduced fit"
        )
    train_drives = _kernel_drives(prepared, train_blocks, state.beta)
    test_drives = _kernel_drives(prepared, test_blocks, state.beta)
    train_design = _gain_design(prepared, train_drives, train_gains)
    test_design = _gain_design(prepared, test_drives, test_gains)
    if train_design is None or test_design is None:
        raise RuntimeError("gain permutation test requires a gain design")
    train_offset = np.full(y_train.size, state.intercept)
    test_offset = np.full(y_test.size, state.intercept)
    for predictor in prepared.spec.predictors:
        if not predictor.gains:
            train_offset += train_drives[predictor.name]
            test_offset += test_drives[predictor.name]
    tested = np.zeros(train_design.shape[1], dtype=bool)
    tested[[prepared.layout.gain_coefficients[term] for term in gain_terms]] = True
    train = _GainStatistics(
        train_design[:, ~tested],
        np.column_stack([train_drives[predictor] for _, predictor in gain_terms]),
        y_train - train_offset,
        train_trials,
    )
    test = _GainStatistics(
        test_design[:, ~tested],
        np.column_stack([test_drives[predictor] for _, predictor in gain_terms]),
        y_test - test_offset,
        test_trials,
    )
    return _ridge_held_out_sse(
        train, test, trial_values, alpha, np.ones(train_design.shape[1])
    )


def predictor_label_sse(
    prepared: PreparedDesign,
    predictor: str,
    trial_values: np.ndarray,
    state: FitState,
    alpha: float,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_blocks: Mapping[str, np.ndarray],
    test_blocks: Mapping[str, np.ndarray],
    train_gains: Mapping[str, np.ndarray],
    test_gains: Mapping[str, np.ndarray],
    train_trials: np.ndarray,
    test_trials: np.ndarray,
) -> np.ndarray:
    """Held-out SSE per label row when kernels are refit with gains frozen.

    Matches ``_fixed_gain_shift_predictions`` for Ridge: the tested one-column
    predictor takes the label value on every row of a trial and all kernels are
    refit with an unpenalized intercept at ``alpha``.
    """
    train_design = _kernel_design(prepared, train_blocks, train_gains, state.gain)
    test_design = _kernel_design(prepared, test_blocks, test_gains, state.gain)
    tested = np.zeros(train_design.shape[1], dtype=bool)
    tested[prepared.layout.beta_slices[predictor]] = True
    train = _PredictorStatistics(train_design[:, ~tested], y_train, train_trials)
    test = _PredictorStatistics(test_design[:, ~tested], y_test, test_trials)
    penalty = np.ones(train_design.shape[1] + 1)
    penalty[0] = 0.0
    return _ridge_held_out_sse(train, test, trial_values, alpha, penalty)

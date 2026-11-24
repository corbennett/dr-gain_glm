"""Grouped or trial-held-out evaluation and reduced-model comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ._label_null import gain_label_sse, predictor_label_sse
from ._solver import (
    _fit_gain_coefficients,
    _gain_design,
    _kernel_drives,
    fit_model,
    fit_state,
    gain_keep_mask,
    predict_parameters,
    r2_score,
    refit_gains,
    refit_kernels,
)
from .design import PreparedDesign
from .model import (
    ConvergenceDiagnostics,
    Dropout,
    FitConfig,
    FitState,
    FittedModel,
    ResolvedDropout,
)


@dataclass(frozen=True)
class CVConfig:
    """Configure outer CV, preferring available trial-group labels by default."""

    folds: int = 5
    seed: int | None = None
    gap_history: bool | int = False
    split: str = "auto"

    def __post_init__(self) -> None:
        if self.folds < 2:
            raise ValueError("CV requires at least two folds")
        if isinstance(self.gap_history, int) and self.gap_history < 0:
            raise ValueError("gap_history cannot be negative")
        if self.split not in {"auto", "groups", "trials"}:
            raise ValueError("CV split must be 'auto', 'groups', or 'trials'")


NULL_METHODS = ("shift", "labels")


@dataclass(frozen=True, eq=False)  # holds arrays: compare by identity
class NullConfig:
    """Choose the null distribution of dropouts with ``permutation_test=True``.

    ``method="shift"`` (default) compares the observed alignment with
    non-wrapping linear trial shifts on the common edge-trimmed trials.
    ``method="labels"`` compares it with caller-supplied pseudo trial values,
    for example contexts with randomly placed block boundaries from
    :func:`gain_glm.dynamic_routing.random_block_contexts`, scored on every
    eligible trial. ``labels`` is an ``(n_labels, n_trials)`` array used for
    every permutation-tested dropout, or a mapping from gain name (gain
    dropouts) or predictor name (predictor dropouts) to such an array. Label
    values replace the trial values of the permuted gain source or the
    trial-constant predictor, so they must use the same units (for example
    +1/-1 context).
    """

    method: str = "shift"
    labels: np.ndarray | Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.method not in NULL_METHODS:
            raise ValueError(f"null method must be one of {NULL_METHODS}")
        if self.method == "shift":
            if self.labels is not None:
                raise ValueError("the shift null does not take labels")
            return
        if self.labels is None:
            raise ValueError("the labels null requires labels")
        if isinstance(self.labels, Mapping):
            labels = {
                str(name): _validated_labels(values, name)
                for name, values in self.labels.items()
            }
            if not labels:
                raise ValueError("labels mapping is empty")
        else:
            labels = _validated_labels(self.labels, "labels")
        object.__setattr__(self, "labels", labels)

    def labels_for(self, name: str, n_trials: int) -> np.ndarray:
        """Return the label matrix used for gain or predictor ``name``."""
        labels = self.labels
        if isinstance(labels, Mapping):
            if name not in labels:
                raise ValueError(f"null labels have no entry for {name!r}")
            labels = labels[name]
        if labels.shape[1] != n_trials:
            raise ValueError(
                f"null labels have {labels.shape[1]} trials, expected {n_trials}"
            )
        return labels


def _validated_labels(values, name) -> np.ndarray:
    labels = np.array(values, dtype=float)
    if labels.ndim != 2 or labels.shape[0] < 1:
        raise ValueError(f"null labels {name!r} must have shape (n_labels, n_trials)")
    if not np.all(np.isfinite(labels)):
        raise ValueError(f"null labels {name!r} contain non-finite values")
    labels.setflags(write=False)
    return labels


@dataclass(frozen=True)
class CVResult:
    r2_per_fold: np.ndarray
    n_iter_per_fold: np.ndarray
    diagnostics_per_fold: tuple[ConvergenceDiagnostics, ...]
    r2_pooled: float = float("nan")
    split: str = "trials"
    held_out_groups: np.ndarray | None = None

    @property
    def r2(self) -> float:
        return float(np.nanmean(self.r2_per_fold))

    @property
    def converged_per_fold(self) -> np.ndarray:
        return np.asarray(
            [diagnostics.converged for diagnostics in self.diagnostics_per_fold],
            dtype=bool,
        )

    @property
    def converged(self) -> bool:
        return bool(np.all(self.converged_per_fold))


@dataclass(frozen=True)
class PermutationTestResult:
    """Frozen-parameter permutation test.

    With ``method="shift"``, ``shifts`` are trial lags on the common
    non-wrapping support. With ``method="labels"``, ``shifts`` are label
    indices: 0 is the observed alignment and ``k`` is null label ``k - 1``,
    scored on every eligible trial. Either way the first entry is observed and
    ``p_value`` is ``(1 + #{null >= observed}) / (1 + n_null)``.
    """

    kind: str
    shifts: np.ndarray
    eligible_trials: np.ndarray
    valid_trials: np.ndarray
    delta_r2_per_shift_per_fold: np.ndarray
    delta_r2_pooled_per_shift: np.ndarray
    reduced_r2_per_fold: np.ndarray
    reduced_r2_pooled: float
    reduced_diagnostics_per_fold: tuple[ConvergenceDiagnostics, ...]
    method: str = "shift"

    @property
    def delta_r2_per_shift(self) -> np.ndarray:
        return np.nanmean(self.delta_r2_per_shift_per_fold, axis=1)

    @property
    def observed_delta_r2(self) -> float:
        return float(self.delta_r2_per_shift[0])

    @property
    def observed_delta_r2_pooled(self) -> float:
        return float(self.delta_r2_pooled_per_shift[0])

    @property
    def p_value(self) -> float:
        values = np.asarray(self.delta_r2_pooled_per_shift, dtype=float)
        finite = np.isfinite(values)
        if not finite[0]:
            return float("nan")
        return float(np.mean(values[finite] >= values[0]))

    @property
    def omitted_trials(self) -> np.ndarray:
        """Eligible edge trials excluded from every shifted comparison."""
        return np.setdiff1d(self.eligible_trials, self.valid_trials, assume_unique=True)

    @property
    def reduced_converged_per_fold(self) -> np.ndarray:
        return np.asarray(
            [
                diagnostics.converged
                for diagnostics in self.reduced_diagnostics_per_fold
            ],
            dtype=bool,
        )


@dataclass(frozen=True)
class DropoutResult:
    dropout: ResolvedDropout
    reduced_r2_per_fold: np.ndarray
    delta_r2_per_fold: np.ndarray
    reduced_diagnostics_per_fold: tuple[ConvergenceDiagnostics, ...] | None = None
    reduced_r2_pooled: float = float("nan")
    delta_r2_pooled: float = float("nan")
    permutation: PermutationTestResult | None = None

    @property
    def reduced_r2(self) -> float:
        return float(np.nanmean(self.reduced_r2_per_fold))

    @property
    def delta_r2(self) -> float:
        return float(np.nanmean(self.delta_r2_per_fold))

    @property
    def reduced_converged_per_fold(self) -> np.ndarray | None:
        if self.reduced_diagnostics_per_fold is None:
            return None
        return np.asarray(
            [
                diagnostics.converged
                for diagnostics in self.reduced_diagnostics_per_fold
            ],
            dtype=bool,
        )


@dataclass(frozen=True)
class EvaluationResult:
    fit: FittedModel
    train_r2: float
    cv: CVResult
    dropouts: Mapping[str, DropoutResult]

    def to_dict(self) -> dict:
        """Return a serialization-friendly summary, including fitted kernels."""
        cv_converged = self.cv.converged_per_fold

        def permutation_summary(
            result: PermutationTestResult | None,
        ) -> dict | None:
            if result is None:
                return None
            # A labels null has ~1000 entries per dropout; serialize only the
            # pooled curve that defines the p-value (per-fold curves stay on the
            # in-memory result).
            compact = result.method == "labels"
            return {
                "kind": result.kind,
                "method": result.method,
                "shifts": None if compact else result.shifts.copy(),
                "n_unique_shifts": int(result.shifts.size),
                "n_null": int(result.shifts.size - 1),
                "eligible_trials": result.eligible_trials.copy(),
                "valid_trials": result.valid_trials.copy(),
                "omitted_trials": result.omitted_trials,
                "p_value": result.p_value,
                "observed_delta_r2": result.observed_delta_r2,
                "observed_delta_r2_pooled": result.observed_delta_r2_pooled,
                "delta_r2_per_shift": (
                    None if compact else result.delta_r2_per_shift.copy()
                ),
                "delta_r2_per_shift_per_fold": (
                    None if compact else result.delta_r2_per_shift_per_fold.copy()
                ),
                "delta_r2_pooled_per_shift": (result.delta_r2_pooled_per_shift.copy()),
                "reduced_r2_per_fold": result.reduced_r2_per_fold.copy(),
                "reduced_r2_pooled": result.reduced_r2_pooled,
                "reduced_converged_per_fold": (result.reduced_converged_per_fold),
                "reduced_convergence_per_fold": [
                    diagnostics.to_dict()
                    for diagnostics in result.reduced_diagnostics_per_fold
                ],
            }

        def dropout_summary(result: DropoutResult) -> dict:
            reduced_converged = result.reduced_converged_per_fold
            metric_converged = cv_converged.copy()
            if reduced_converged is not None:
                metric_converged &= reduced_converged
            return {
                "predictors": result.dropout.predictors,
                "gains": result.dropout.gains,
                "gain_terms": result.dropout.gain_terms,
                "refit": result.dropout.refit,
                "permutation_test": result.dropout.permutation_test,
                "reduced_r2": result.reduced_r2,
                "reduced_r2_per_fold": result.reduced_r2_per_fold.copy(),
                "reduced_r2_pooled": result.reduced_r2_pooled,
                "delta_r2": result.delta_r2,
                "delta_r2_per_fold": result.delta_r2_per_fold.copy(),
                "delta_r2_pooled": result.delta_r2_pooled,
                "metric_converged": bool(np.all(metric_converged)),
                "metric_converged_per_fold": metric_converged,
                "reduced_converged_per_fold": reduced_converged,
                "reduced_convergence_per_fold": (
                    None
                    if result.reduced_diagnostics_per_fold is None
                    else [
                        diagnostics.to_dict()
                        for diagnostics in result.reduced_diagnostics_per_fold
                    ]
                ),
                "permutation": permutation_summary(result.permutation),
            }

        return {
            "model": self.fit.spec.name,
            "train_r2": self.train_r2,
            "cv_r2": self.cv.r2,
            "cv_r2_per_fold": self.cv.r2_per_fold.copy(),
            "cv_r2_pooled": self.cv.r2_pooled,
            "cv_split": self.cv.split,
            "cv_held_out_groups": (
                None
                if self.cv.held_out_groups is None
                else self.cv.held_out_groups.copy()
            ),
            "n_iter": self.fit.n_iter,
            "n_iter_per_fold": self.cv.n_iter_per_fold.copy(),
            # ``converged`` is retained for compatibility and describes only
            # the final all-data fit. CV-derived metrics use the fields below.
            "converged": self.fit.converged,
            "final_converged": self.fit.converged,
            "final_convergence": self.fit.state.diagnostics.to_dict(),
            "cv_converged": self.cv.converged,
            "cv_converged_per_fold": cv_converged,
            "cv_convergence_per_fold": [
                diagnostics.to_dict() for diagnostics in self.cv.diagnostics_per_fold
            ],
            "intercept": self.fit.intercept,
            "kernels": self.fit.kernels(),
            "gain_table": self.fit.gain_table(),
            "dropouts": {
                name: dropout_summary(result) for name, result in self.dropouts.items()
            },
        }


def _safe_history_rows(rows: np.ndarray, lag_bins: int) -> np.ndarray:
    if lag_bins <= 0:
        return rows.copy()
    safe = rows.copy()
    for lag in range(1, lag_bins + 1):
        if lag >= rows.size:
            return np.zeros_like(rows)
        safe[:lag] = False
        safe[lag:] &= rows[:-lag]
    return safe


def _evaluation_mask(prepared: PreparedDesign, mask: np.ndarray | None) -> np.ndarray:
    selected = (
        prepared.fit_mask if mask is None else np.asarray(mask, dtype=bool).ravel()
    )
    if selected.size != prepared.data.n_time:
        raise ValueError(
            f"mask has length {selected.size}, expected {prepared.data.n_time}"
        )
    if not selected.any():
        raise ValueError("mask selects no bins")
    return selected


def _r2_from_sse(target: np.ndarray, squared_error: float) -> float:
    """Score pooled held-out residuals against one global target mean."""
    total = float(np.sum((target - target.mean()) ** 2))
    if total <= 0:
        return float("nan")
    return 1 - squared_error / total


def _outer_folds(
    prepared: PreparedDesign,
    selected: np.ndarray,
    config: CVConfig,
) -> tuple[list[np.ndarray], str, np.ndarray | None]:
    """Return test-trial arrays and metadata for the requested outer split."""
    trials = np.unique(prepared.data.trial_index[selected])
    use_groups = config.split == "groups" or (
        config.split == "auto" and prepared.data.cv_groups is not None
    )
    if use_groups:
        if prepared.data.cv_groups is None:
            raise ValueError("grouped CV requires data.cv_groups")
        trial_groups = prepared.data.cv_groups[trials]
        _, first_indices = np.unique(trial_groups, return_index=True)
        groups = trial_groups[np.sort(first_indices)]
        if groups.size < 2:
            raise ValueError(
                "grouped CV requires at least two groups with selected rows"
            )
        folds = [trials[trial_groups == group] for group in groups]
        return folds, "groups", groups

    if config.folds > trials.size:
        raise ValueError(
            f"CV requests {config.folds} folds for only {trials.size} trials"
        )
    if config.seed is not None:
        trials = np.random.default_rng(config.seed).permutation(trials)
    return list(np.array_split(trials, config.folds)), "trials", None


@dataclass(frozen=True)
class _PermutationPlan:
    kind: str
    shifts: np.ndarray
    trial_values_per_shift: np.ndarray
    eligible_trials: np.ndarray
    valid_trials: np.ndarray
    gain_name: str | None = None
    gain_terms: tuple[tuple[str, str], ...] = ()
    predictor: str | None = None
    method: str = "shift"


def _edge_block_shifts(values: np.ndarray) -> np.ndarray:
    """Return lags supported by half of each contiguous edge-value block.

    Positive ``np.roll`` shifts consume support at the beginning of the trial
    sequence, so their limit comes from the first block. Negative shifts
    consume support at the end and use the last block's limit. Shift zero is
    kept first as the observed alignment.
    """
    trial_values = np.asarray(values, dtype=float).ravel()
    if trial_values.size < 2:
        raise ValueError("a permutation test requires at least two selected trials")

    first_change = np.flatnonzero(~np.isclose(trial_values, trial_values[0]))
    if first_change.size == 0:
        raise ValueError("a permutation test requires at least two trial-value blocks")
    first_block_length = int(first_change[0])

    last_change = np.flatnonzero(~np.isclose(trial_values, trial_values[-1]))
    last_block_length = trial_values.size - int(last_change[-1]) - 1

    max_positive = first_block_length // 2
    max_negative = last_block_length // 2
    if max_positive + max_negative == 0:
        raise ValueError(
            "permutation test edge blocks are too short for a nonzero shift"
        )
    return np.concatenate(
        (
            np.arange(max_positive + 1, dtype=int),
            np.arange(-max_negative, 0, dtype=int),
        )
    )


def _common_nonwrapping_trials(
    active_trials: np.ndarray, shifts: np.ndarray
) -> np.ndarray:
    """Return trials whose shifted source index is in bounds for every lag."""
    first = max(0, int(np.max(shifts)))
    stop = active_trials.size + min(0, int(np.min(shifts)))
    valid_trials = active_trials[first:stop]
    if valid_trials.size < 2:
        raise ValueError(
            "permutation test has fewer than two trials in its common "
            "edge-trimmed window"
        )
    return valid_trials


def _gain_terms_for_permutation(
    prepared: PreparedDesign, dropout: ResolvedDropout
) -> tuple[str, tuple[tuple[str, str], ...]]:
    requested_gains = set(dropout.gains)
    requested_terms = set(dropout.gain_terms)
    gain_names = requested_gains | {gain for gain, _ in requested_terms}
    if len(gain_names) != 1:
        raise ValueError(
            f"dropout {dropout.name!r}: a gain permutation test must select "
            "exactly one gain"
        )
    gain_name = next(iter(gain_names))
    terms = tuple(
        term
        for term in prepared.layout.gain_coefficients
        if (term[0] in requested_gains or term in requested_terms)
    )
    if not terms:
        raise ValueError(
            f"dropout {dropout.name!r} selects no gain coefficients to permute"
        )
    return gain_name, terms


def _label_rows(
    labels: np.ndarray,
    active_trials: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    """Stack the observed trial values and the labels on the active trials."""
    rows = np.tile(base, (labels.shape[0] + 1, 1))
    rows[1:, active_trials] = labels[:, active_trials]
    return rows


def _permutation_plan(
    prepared: PreparedDesign,
    dropout: ResolvedDropout,
    selected: np.ndarray,
    null: NullConfig | None = None,
) -> _PermutationPlan:
    use_labels = null is not None and null.method == "labels"
    active_trials = np.unique(prepared.data.trial_index[selected])
    n_trials = prepared.data.n_trials
    if dropout.predictors:
        predictor = dropout.predictors[0]
        if prepared.spec.predictor(predictor).gains:
            raise ValueError(
                f"dropout {dropout.name!r}: predictor permutation tests require "
                "a predictor without gains"
            )
        if predictor not in prepared.base_blocks:
            raise ValueError(
                f"dropout {dropout.name!r}: predictor permutation tests do not "
                "support target-history predictors"
            )
        block = prepared.base_blocks[predictor]
        if block.shape[1] != 1:
            raise ValueError(
                f"dropout {dropout.name!r}: predictor permutation tests require "
                "a one-column design"
            )
        active_values = np.empty(active_trials.size)
        for index, trial in enumerate(active_trials):
            rows = selected & (prepared.data.trial_index == trial)
            values = block[rows, 0]
            if values.size == 0 or not np.allclose(values, values[0]):
                raise ValueError(
                    f"dropout {dropout.name!r}: predictor {predictor!r} must be "
                    "constant within every selected trial"
                )
            active_values[index] = values[0]
        if use_labels:
            labels = null.labels_for(predictor, n_trials)
            observed = np.unique(active_values)
            allowed = np.isclose(
                labels[:, active_trials][..., None], observed[None, None, :]
            ).any(axis=-1)
            if not allowed.all():
                raise ValueError(
                    f"dropout {dropout.name!r}: null labels for predictor "
                    f"{predictor!r} must take observed predictor values "
                    f"{observed.tolist()}"
                )
            base = np.zeros(n_trials)
            base[active_trials] = active_values
            return _PermutationPlan(
                kind="predictor",
                shifts=np.arange(labels.shape[0] + 1),
                trial_values_per_shift=_label_rows(labels, active_trials, base),
                eligible_trials=active_trials,
                valid_trials=active_trials.copy(),
                predictor=predictor,
                method="labels",
            )
        shifts = _edge_block_shifts(active_values)
        valid_trials = _common_nonwrapping_trials(active_trials, shifts)
        shifted = np.zeros((shifts.size, n_trials))
        for index, shift in enumerate(shifts):
            shifted[index, active_trials] = np.roll(active_values, shift)
        return _PermutationPlan(
            kind="predictor",
            shifts=shifts,
            trial_values_per_shift=shifted,
            eligible_trials=active_trials,
            valid_trials=valid_trials,
            predictor=predictor,
        )

    gain_name, gain_terms = _gain_terms_for_permutation(prepared, dropout)
    gain_spec = next(gain for gain in prepared.spec.gains if gain.name == gain_name)
    source_values = np.asarray(
        prepared.data.trial_values[gain_spec.source], dtype=float
    )[:n_trials]
    active_values = source_values[active_trials]
    if use_labels:
        labels = null.labels_for(gain_name, n_trials)
        return _PermutationPlan(
            kind="gain",
            shifts=np.arange(labels.shape[0] + 1),
            trial_values_per_shift=_label_rows(labels, active_trials, source_values),
            eligible_trials=active_trials,
            valid_trials=active_trials.copy(),
            gain_name=gain_name,
            gain_terms=gain_terms,
            method="labels",
        )
    shifts = _edge_block_shifts(active_values)
    valid_trials = _common_nonwrapping_trials(active_trials, shifts)
    shifted = np.tile(source_values, (shifts.size, 1))
    for index, shift in enumerate(shifts):
        shifted[index, active_trials] = np.roll(active_values, shift)
    return _PermutationPlan(
        kind="gain",
        shifts=shifts,
        trial_values_per_shift=shifted,
        eligible_trials=active_trials,
        valid_trials=valid_trials,
        gain_name=gain_name,
        gain_terms=gain_terms,
    )


def _fixed_kernel_shift_predictions(
    prepared: PreparedDesign,
    plan: _PermutationPlan,
    state: FitState,
    y_train: np.ndarray,
    train_blocks: Mapping[str, np.ndarray],
    test_blocks: Mapping[str, np.ndarray],
    train_gains: Mapping[str, np.ndarray],
    test_gains: Mapping[str, np.ndarray],
    train_trials: np.ndarray,
    test_trials: np.ndarray,
    config: FitConfig,
) -> list[np.ndarray]:
    """Refit gains for every shift while holding reduced kernels fixed."""
    train_drives = _kernel_drives(prepared, train_blocks, state.beta)
    test_drives = _kernel_drives(prepared, test_blocks, state.beta)
    train_design = _gain_design(prepared, train_drives, train_gains)
    test_design = _gain_design(prepared, test_drives, test_gains)
    if train_design is None or test_design is None:
        raise RuntimeError("gain permutation test requires a gain design")
    train_design = train_design.copy()
    test_design = test_design.copy()
    train_offset = np.full(y_train.size, state.intercept)
    test_offset = np.full(test_trials.size, state.intercept)
    for predictor in prepared.spec.predictors:
        if predictor.gains:
            continue
        train_offset += train_drives[predictor.name]
        test_offset += test_drives[predictor.name]

    alpha = state.iterations[-1].gain_alpha
    if alpha is None:
        raise RuntimeError(
            "gain permutation test requires a gain penalty from the reduced fit"
        )
    predictions: list[np.ndarray] = []
    for shifted_trial_values in plan.trial_values_per_shift:
        shifted_train = shifted_trial_values[train_trials]
        shifted_test = shifted_trial_values[test_trials]
        for term in plan.gain_terms:
            _, predictor = term
            column = prepared.layout.gain_coefficients[term]
            train_design[:, column] = shifted_train * train_drives[predictor]
            test_design[:, column] = shifted_test * test_drives[predictor]
        shifted_gain, _ = _fit_gain_coefficients(
            prepared,
            config,
            train_design,
            y_train - train_offset,
            trial_index=train_trials,
            alpha=alpha,
        )
        predictions.append(test_offset + test_design @ shifted_gain)
    return predictions


def _fixed_gain_shift_predictions(
    prepared: PreparedDesign,
    plan: _PermutationPlan,
    state: FitState,
    y_train: np.ndarray,
    train_blocks: Mapping[str, np.ndarray],
    test_blocks: Mapping[str, np.ndarray],
    train_gains: Mapping[str, np.ndarray],
    test_gains: Mapping[str, np.ndarray],
    train_trials: np.ndarray,
    test_trials: np.ndarray,
    config: FitConfig,
) -> list[np.ndarray]:
    """Refit kernels for every shift while holding reduced gains fixed."""
    if plan.predictor is None:
        raise RuntimeError("predictor permutation plan has no predictor")
    alpha = state.iterations[-1].kernel_alpha
    predictions: list[np.ndarray] = []
    for shifted_trial_values in plan.trial_values_per_shift:
        shifted_train_blocks = dict(train_blocks)
        shifted_test_blocks = dict(test_blocks)
        shifted_train_blocks[plan.predictor] = shifted_trial_values[train_trials][
            :, None
        ]
        shifted_test_blocks[plan.predictor] = shifted_trial_values[test_trials][:, None]
        beta, intercept, _ = refit_kernels(
            prepared,
            y_train,
            shifted_train_blocks,
            train_gains,
            state.gain,
            config,
            trial_index=train_trials,
            alpha=alpha,
        )
        predictions.append(
            predict_parameters(
                prepared,
                shifted_test_blocks,
                test_gains,
                beta,
                state.gain,
                intercept,
            )
        )
    return predictions


_BATCHED_LABEL_NULL = True


def _batched_label_sse(
    prepared: PreparedDesign,
    plan: _PermutationPlan,
    state: FitState,
    config: FitConfig,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_blocks: Mapping[str, np.ndarray],
    test_blocks: Mapping[str, np.ndarray],
    train_gains: Mapping[str, np.ndarray],
    test_gains: Mapping[str, np.ndarray],
    train_trials: np.ndarray,
    test_trials: np.ndarray,
) -> np.ndarray | None:
    """Score every null label at once when the linear subproblem is Ridge.

    Returns ``None`` when the per-label refits must be used instead: the shift
    null (kept on its original code path), Lasso kernels, or a kernel penalty
    that would have to be selected by inner CV.
    """
    if (
        plan.method != "labels"
        or not _BATCHED_LABEL_NULL
        or config.regularizer != "ridge"
    ):
        return None
    if plan.kind == "gain":
        return gain_label_sse(
            prepared,
            plan.gain_terms,
            plan.trial_values_per_shift,
            state,
            y_train,
            y_test,
            train_blocks,
            test_blocks,
            train_gains,
            test_gains,
            train_trials,
            test_trials,
        )
    alpha = state.iterations[-1].kernel_alpha
    if alpha is None:
        alpha = config.kernel_alpha
    if alpha is None or plan.predictor is None:
        return None
    return predictor_label_sse(
        prepared,
        plan.predictor,
        plan.trial_values_per_shift,
        state,
        alpha,
        y_train,
        y_test,
        train_blocks,
        test_blocks,
        train_gains,
        test_gains,
        train_trials,
        test_trials,
    )


def evaluate(
    prepared: PreparedDesign,
    y: np.ndarray,
    *,
    fit: FitConfig | None = None,
    cv: CVConfig | None = None,
    dropouts: Sequence[Dropout] | None = None,
    mask: np.ndarray | None = None,
    null: NullConfig | None = None,
) -> EvaluationResult:
    """Fit once and evaluate all requested dropouts in one shared CV pass.

    ``null`` selects the permutation-test null (default: linear shifts).
    """
    values = np.asarray(y, dtype=float).ravel()
    if values.size != prepared.data.n_time:
        raise ValueError(
            f"target has length {values.size}, expected {prepared.data.n_time}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("target contains non-finite values")
    fit_config = FitConfig() if fit is None else fit
    cv_config = CVConfig() if cv is None else cv
    selected = _evaluation_mask(prepared, mask)

    requests = prepared.spec.dropouts if dropouts is None else tuple(dropouts)
    names = [dropout.name for dropout in requests]
    if len(set(names)) != len(names):
        raise ValueError("dropout names must be unique")
    resolved = tuple(dropout.resolve(prepared) for dropout in requests)
    permutation_plans = {
        dropout.name: _permutation_plan(prepared, dropout, selected, null)
        for dropout in resolved
        if dropout.permutation_test
    }

    fitted = fit_model(prepared, values, config=fit_config, mask=selected)
    train_r2 = fitted.score(values, selected)
    blocks = prepared.blocks_for_target(values if prepared.has_history else None)

    # A row mask may exclude complete trials (for example instruction trials),
    # so do not create outer folds from trials or groups with no selected rows.
    folds, cv_split, held_out_groups = _outer_folds(prepared, selected, cv_config)

    history_gap = 0
    if cv_config.gap_history:
        if not prepared.has_history:
            raise ValueError("gap_history requires a History predictor")
        history_gap = (
            prepared.history_lag_bins
            if cv_config.gap_history is True
            else int(cv_config.gap_history)
        )

    full_r2: list[float] = []
    pooled_targets: list[np.ndarray] = []
    full_squared_error = 0.0
    full_iterations: list[int] = []
    full_diagnostics: list[ConvergenceDiagnostics] = []
    reduced_r2: dict[str, list[float]] = {dropout.name: [] for dropout in resolved}
    delta_r2: dict[str, list[float]] = {dropout.name: [] for dropout in resolved}
    reduced_squared_error: dict[str, float] = {
        dropout.name: 0.0 for dropout in resolved
    }
    reduced_diagnostics: dict[str, list[ConvergenceDiagnostics]] = {
        dropout.name: [] for dropout in resolved
    }
    permutation_reduced_r2: dict[str, list[float]] = {
        name: [] for name in permutation_plans
    }
    permutation_pooled_targets: dict[str, list[np.ndarray]] = {
        name: [] for name in permutation_plans
    }
    permutation_delta_r2: dict[str, list[np.ndarray]] = {
        name: [] for name in permutation_plans
    }
    permutation_reduced_squared_error = {name: 0.0 for name in permutation_plans}
    permutation_shifted_squared_error = {
        name: np.zeros(plan.shifts.size) for name, plan in permutation_plans.items()
    }
    permutation_reduced_diagnostics: dict[str, list[ConvergenceDiagnostics]] = {
        name: [] for name in permutation_plans
    }

    for fold_number, test_trials in enumerate(folds, start=1):
        train_rows = np.isin(prepared.data.trial_index, test_trials, invert=True)
        test_rows = np.isin(prepared.data.trial_index, test_trials)
        if history_gap:
            train_rows = _safe_history_rows(train_rows, history_gap)
            test_rows = _safe_history_rows(test_rows, history_gap)
        train_rows &= selected
        test_rows &= selected
        if not train_rows.any() or not test_rows.any():
            raise ValueError(f"fold {fold_number} has no selected train or test rows")

        train_blocks = {name: block[train_rows] for name, block in blocks.items()}
        test_blocks = {name: block[test_rows] for name, block in blocks.items()}
        train_gains = {
            name: values_by_time[train_rows]
            for name, values_by_time in prepared.gain_by_time.items()
        }
        test_gains = {
            name: values_by_time[test_rows]
            for name, values_by_time in prepared.gain_by_time.items()
        }
        full_state = fit_state(
            prepared,
            values[train_rows],
            train_blocks,
            train_gains,
            fit_config,
            trial_index=prepared.data.trial_index[train_rows],
        )
        full_prediction = predict_parameters(
            prepared,
            test_blocks,
            test_gains,
            full_state.beta,
            full_state.gain,
            full_state.intercept,
        )
        fold_target = values[test_rows]
        fold_full_r2 = r2_score(fold_target, full_prediction)
        full_r2.append(fold_full_r2)
        pooled_targets.append(fold_target)
        full_squared_error += float(np.sum((fold_target - full_prediction) ** 2))
        full_iterations.append(len(full_state.iterations))
        full_diagnostics.append(full_state.diagnostics)

        for dropout in resolved:
            keep = gain_keep_mask(
                prepared,
                remove_gains=dropout.gains,
                remove_gain_terms=dropout.gain_terms,
                remove_predictors=dropout.predictors,
            )
            if dropout.refit == "gains":
                full_gain_alpha = full_state.iterations[-1].gain_alpha
                if full_gain_alpha is None:
                    raise RuntimeError(
                        "gain-only dropout requires a gain penalty from the full fit"
                    )
                reduced_gain, _ = refit_gains(
                    prepared,
                    values[train_rows],
                    train_blocks,
                    train_gains,
                    full_state.beta,
                    full_state.intercept,
                    fit_config,
                    trial_index=prepared.data.trial_index[train_rows],
                    keep=keep,
                    alpha=full_gain_alpha,
                )
                reduced_beta = full_state.beta
                reduced_intercept = full_state.intercept
                reduced_test_blocks = test_blocks
            else:
                removed = set(dropout.predictors)
                reduced_train_blocks = {
                    name: np.zeros_like(block) if name in removed else block
                    for name, block in train_blocks.items()
                }
                reduced_state = fit_state(
                    prepared,
                    values[train_rows],
                    reduced_train_blocks,
                    train_gains,
                    fit_config,
                    trial_index=prepared.data.trial_index[train_rows],
                    keep_gains=keep,
                )
                reduced_beta = reduced_state.beta
                reduced_gain = reduced_state.gain
                reduced_intercept = reduced_state.intercept
                reduced_diagnostics[dropout.name].append(reduced_state.diagnostics)
                reduced_test_blocks = {
                    name: np.zeros_like(block) if name in removed else block
                    for name, block in test_blocks.items()
                }

            reduced_prediction = predict_parameters(
                prepared,
                reduced_test_blocks,
                test_gains,
                reduced_beta,
                reduced_gain,
                reduced_intercept,
            )
            fold_reduced_r2 = r2_score(fold_target, reduced_prediction)
            reduced_r2[dropout.name].append(fold_reduced_r2)
            delta_r2[dropout.name].append(fold_full_r2 - fold_reduced_r2)
            reduced_squared_error[dropout.name] += float(
                np.sum((fold_target - reduced_prediction) ** 2)
            )

            if not dropout.permutation_test:
                continue
            plan = permutation_plans[dropout.name]
            valid_rows = np.isin(prepared.data.trial_index, plan.valid_trials)
            permutation_train_rows = train_rows & valid_rows
            permutation_test_rows = test_rows & valid_rows
            if not permutation_train_rows.any() or not permutation_test_rows.any():
                raise ValueError(
                    f"fold {fold_number} has no common non-wrapping "
                    f"train or test rows for permutation dropout {dropout.name!r}"
                )

            permutation_train_blocks = {
                name: block[permutation_train_rows] for name, block in blocks.items()
            }
            permutation_test_blocks = {
                name: block[permutation_test_rows] for name, block in blocks.items()
            }
            permutation_train_gains = {
                name: values_by_time[permutation_train_rows]
                for name, values_by_time in prepared.gain_by_time.items()
            }
            permutation_test_gains = {
                name: values_by_time[permutation_test_rows]
                for name, values_by_time in prepared.gain_by_time.items()
            }
            removed = set(dropout.predictors)
            permutation_reduced_train_blocks = {
                name: np.zeros_like(block) if name in removed else block
                for name, block in permutation_train_blocks.items()
            }
            permutation_reduced_test_blocks = {
                name: np.zeros_like(block) if name in removed else block
                for name, block in permutation_test_blocks.items()
            }
            permutation_train_trials = prepared.data.trial_index[permutation_train_rows]
            permutation_test_trials = prepared.data.trial_index[permutation_test_rows]
            permutation_state = fit_state(
                prepared,
                values[permutation_train_rows],
                permutation_reduced_train_blocks,
                permutation_train_gains,
                fit_config,
                trial_index=permutation_train_trials,
                keep_gains=keep,
            )
            permutation_reduced_diagnostics[dropout.name].append(
                permutation_state.diagnostics
            )
            permutation_reduced_prediction = predict_parameters(
                prepared,
                permutation_reduced_test_blocks,
                permutation_test_gains,
                permutation_state.beta,
                permutation_state.gain,
                permutation_state.intercept,
            )
            permutation_target = values[permutation_test_rows]
            fold_permutation_reduced_r2 = r2_score(
                permutation_target, permutation_reduced_prediction
            )
            permutation_reduced_r2[dropout.name].append(fold_permutation_reduced_r2)
            permutation_pooled_targets[dropout.name].append(permutation_target)
            permutation_reduced_squared_error[dropout.name] += float(
                np.sum((permutation_target - permutation_reduced_prediction) ** 2)
            )

            label_sse = _batched_label_sse(
                prepared,
                plan,
                permutation_state,
                fit_config,
                values[permutation_train_rows],
                permutation_target,
                permutation_train_blocks,
                permutation_test_blocks,
                permutation_train_gains,
                permutation_test_gains,
                permutation_train_trials,
                permutation_test_trials,
            )
            if label_sse is not None:
                total = float(
                    np.sum((permutation_target - permutation_target.mean()) ** 2)
                )
                shifted_r2 = (
                    1 - label_sse / total
                    if total > 0
                    else np.full(label_sse.size, np.nan)
                )
                permutation_delta_r2[dropout.name].append(
                    shifted_r2 - fold_permutation_reduced_r2
                )
                permutation_shifted_squared_error[dropout.name] += label_sse
                continue
            if plan.kind == "gain":
                shifted_predictions = _fixed_kernel_shift_predictions(
                    prepared,
                    plan,
                    permutation_state,
                    values[permutation_train_rows],
                    permutation_train_blocks,
                    permutation_test_blocks,
                    permutation_train_gains,
                    permutation_test_gains,
                    permutation_train_trials,
                    permutation_test_trials,
                    fit_config,
                )
            else:
                shifted_predictions = _fixed_gain_shift_predictions(
                    prepared,
                    plan,
                    permutation_state,
                    values[permutation_train_rows],
                    permutation_train_blocks,
                    permutation_test_blocks,
                    permutation_train_gains,
                    permutation_test_gains,
                    permutation_train_trials,
                    permutation_test_trials,
                    fit_config,
                )
            shifted_r2 = np.asarray(
                [
                    r2_score(permutation_target, prediction)
                    for prediction in shifted_predictions
                ]
            )
            permutation_delta_r2[dropout.name].append(
                shifted_r2 - fold_permutation_reduced_r2
            )
            permutation_shifted_squared_error[dropout.name] += np.asarray(
                [
                    np.sum((permutation_target - prediction) ** 2)
                    for prediction in shifted_predictions
                ]
            )

    pooled_target = np.concatenate(pooled_targets)
    full_r2_pooled = _r2_from_sse(pooled_target, full_squared_error)
    reduced_r2_pooled = {
        dropout.name: _r2_from_sse(pooled_target, reduced_squared_error[dropout.name])
        for dropout in resolved
    }
    permutation_results: dict[str, PermutationTestResult] = {}
    for name, plan in permutation_plans.items():
        permutation_pooled_target = np.concatenate(permutation_pooled_targets[name])
        permutation_reduced_pooled = _r2_from_sse(
            permutation_pooled_target, permutation_reduced_squared_error[name]
        )
        shifted_r2_pooled = np.asarray(
            [
                _r2_from_sse(permutation_pooled_target, squared_error)
                for squared_error in permutation_shifted_squared_error[name]
            ]
        )
        permutation_results[name] = PermutationTestResult(
            kind=plan.kind,
            shifts=plan.shifts.copy(),
            eligible_trials=plan.eligible_trials.copy(),
            valid_trials=plan.valid_trials.copy(),
            delta_r2_per_shift_per_fold=np.asarray(
                permutation_delta_r2[name], dtype=float
            ).T,
            delta_r2_pooled_per_shift=(shifted_r2_pooled - permutation_reduced_pooled),
            reduced_r2_per_fold=np.asarray(permutation_reduced_r2[name], dtype=float),
            reduced_r2_pooled=permutation_reduced_pooled,
            reduced_diagnostics_per_fold=tuple(permutation_reduced_diagnostics[name]),
            method=plan.method,
        )

    dropout_results = {
        dropout.name: DropoutResult(
            dropout=dropout,
            reduced_r2_per_fold=np.asarray(reduced_r2[dropout.name], dtype=float),
            delta_r2_per_fold=np.asarray(delta_r2[dropout.name], dtype=float),
            reduced_r2_pooled=reduced_r2_pooled[dropout.name],
            delta_r2_pooled=(full_r2_pooled - reduced_r2_pooled[dropout.name]),
            reduced_diagnostics_per_fold=(
                tuple(reduced_diagnostics[dropout.name])
                if dropout.refit == "full"
                else None
            ),
            permutation=permutation_results.get(dropout.name),
        )
        for dropout in resolved
    }
    return EvaluationResult(
        fit=fitted,
        train_r2=train_r2,
        cv=CVResult(
            r2_per_fold=np.asarray(full_r2, dtype=float),
            r2_pooled=full_r2_pooled,
            n_iter_per_fold=np.asarray(full_iterations, dtype=int),
            diagnostics_per_fold=tuple(full_diagnostics),
            split=cv_split,
            held_out_groups=held_out_groups,
        ),
        dropouts=dropout_results,
    )

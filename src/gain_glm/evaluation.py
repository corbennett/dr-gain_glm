"""Trial-held-out evaluation and declarative reduced-model comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ._solver import (
    fit_model,
    fit_state,
    gain_keep_mask,
    predict_parameters,
    r2_score,
    refit_gains,
)
from .design import PreparedDesign
from .model import (
    ConvergenceDiagnostics,
    Dropout,
    FitConfig,
    FittedModel,
    ResolvedDropout,
)


@dataclass(frozen=True)
class CVConfig:
    folds: int = 5
    seed: int | None = None
    gap_history: bool | int = False

    def __post_init__(self) -> None:
        if self.folds < 2:
            raise ValueError("CV requires at least two folds")
        if isinstance(self.gap_history, int) and self.gap_history < 0:
            raise ValueError("gap_history cannot be negative")


@dataclass(frozen=True)
class CVResult:
    r2_per_fold: np.ndarray
    n_iter_per_fold: np.ndarray
    diagnostics_per_fold: tuple[ConvergenceDiagnostics, ...]
    r2_pooled: float = float("nan")

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
class DropoutResult:
    dropout: ResolvedDropout
    reduced_r2_per_fold: np.ndarray
    delta_r2_per_fold: np.ndarray
    reduced_diagnostics_per_fold: tuple[ConvergenceDiagnostics, ...] | None = None
    reduced_r2_pooled: float = float("nan")
    delta_r2_pooled: float = float("nan")

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
            }

        return {
            "model": self.fit.spec.name,
            "train_r2": self.train_r2,
            "cv_r2": self.cv.r2,
            "cv_r2_per_fold": self.cv.r2_per_fold.copy(),
            "cv_r2_pooled": self.cv.r2_pooled,
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
                diagnostics.to_dict()
                for diagnostics in self.cv.diagnostics_per_fold
            ],
            "intercept": self.fit.intercept,
            "kernels": self.fit.kernels(),
            "gain_table": self.fit.gain_table(),
            "dropouts": {
                name: dropout_summary(result)
                for name, result in self.dropouts.items()
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


def evaluate(
    prepared: PreparedDesign,
    y: np.ndarray,
    *,
    fit: FitConfig | None = None,
    cv: CVConfig | None = None,
    dropouts: Sequence[Dropout] | None = None,
    mask: np.ndarray | None = None,
) -> EvaluationResult:
    """Fit once and evaluate all requested dropouts in one shared CV pass."""
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

    fitted = fit_model(prepared, values, config=fit_config, mask=selected)
    train_r2 = fitted.score(values, selected)
    blocks = prepared.blocks_for_target(values if prepared.has_history else None)

    trials = np.unique(prepared.data.trial_index)
    if cv_config.folds > trials.size:
        raise ValueError(
            f"CV requests {cv_config.folds} folds for only {trials.size} trials"
        )
    if cv_config.seed is not None:
        trials = np.random.default_rng(cv_config.seed).permutation(trials)
    folds = np.array_split(trials, cv_config.folds)

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
                    keep_gains=keep,
                )
                reduced_beta = reduced_state.beta
                reduced_gain = reduced_state.gain
                reduced_intercept = reduced_state.intercept
                reduced_diagnostics[dropout.name].append(
                    reduced_state.diagnostics
                )
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

    pooled_target = np.concatenate(pooled_targets)
    full_r2_pooled = _r2_from_sse(pooled_target, full_squared_error)
    reduced_r2_pooled = {
        dropout.name: _r2_from_sse(
            pooled_target, reduced_squared_error[dropout.name]
        )
        for dropout in resolved
    }

    dropout_results = {
        dropout.name: DropoutResult(
            dropout=dropout,
            reduced_r2_per_fold=np.asarray(
                reduced_r2[dropout.name], dtype=float
            ),
            delta_r2_per_fold=np.asarray(delta_r2[dropout.name], dtype=float),
            reduced_r2_pooled=reduced_r2_pooled[dropout.name],
            delta_r2_pooled=(
                full_r2_pooled - reduced_r2_pooled[dropout.name]
            ),
            reduced_diagnostics_per_fold=(
                tuple(reduced_diagnostics[dropout.name])
                if dropout.refit == "full"
                else None
            ),
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
        ),
        dropouts=dropout_results,
    )

"""Trial-held-out evaluation and declarative reduced-model comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

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
from .model import FitConfig, FittedModel


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
class Dropout:
    """A named request to remove gains, predictors, or predictor groups."""

    name: str
    remove_predictors: tuple[str, ...] = ()
    remove_gains: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    refit: Literal["auto", "gains", "full"] = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "remove_predictors", tuple(self.remove_predictors))
        object.__setattr__(self, "remove_gains", tuple(self.remove_gains))
        object.__setattr__(self, "groups", tuple(self.groups))
        if not self.name:
            raise ValueError("dropout names cannot be empty")
        if self.refit not in {"auto", "gains", "full"}:
            raise ValueError("refit must be 'auto', 'gains', or 'full'")

    @classmethod
    def gain(
        cls,
        gain: str,
        *,
        name: str | None = None,
        refit: Literal["auto", "gains", "full"] = "auto",
    ) -> Dropout:
        return cls(name or gain, remove_gains=(gain,), refit=refit)

    @classmethod
    def predictors(
        cls,
        *predictors: str,
        name: str | None = None,
        refit: Literal["auto", "full"] = "auto",
    ) -> Dropout:
        label = name or "+".join(predictors)
        return cls(label, remove_predictors=tuple(predictors), refit=refit)

    @classmethod
    def group(
        cls,
        group: str,
        *,
        name: str | None = None,
        refit: Literal["auto", "full"] = "auto",
    ) -> Dropout:
        return cls(name or group, groups=(group,), refit=refit)

    @classmethod
    def terms(
        cls,
        name: str,
        *,
        predictors: Sequence[str] = (),
        gains: Sequence[str] = (),
        groups: Sequence[str] = (),
        refit: Literal["auto", "gains", "full"] = "auto",
    ) -> Dropout:
        """Declare a mixed reduction using any combination of term selectors."""
        return cls(
            name,
            remove_predictors=tuple(predictors),
            remove_gains=tuple(gains),
            groups=tuple(groups),
            refit=refit,
        )

    def resolve(self, prepared: PreparedDesign) -> ResolvedDropout:
        known_predictors = set(prepared.spec.predictor_names)
        known_gains = set(prepared.spec.gain_names)
        unknown_predictors = set(self.remove_predictors) - known_predictors
        unknown_gains = set(self.remove_gains) - known_gains
        if unknown_predictors:
            raise ValueError(
                f"dropout {self.name!r} has unknown predictors: "
                f"{sorted(unknown_predictors)}"
            )
        if unknown_gains:
            raise ValueError(
                f"dropout {self.name!r} has unknown gains: {sorted(unknown_gains)}"
            )

        predictors = set(self.remove_predictors)
        for group in self.groups:
            matches = set(prepared.spec.group_members(group))
            if not matches:
                raise ValueError(
                    f"dropout {self.name!r}: no predictors belong to group {group!r}"
                )
            predictors.update(matches)
        if not predictors and not self.remove_gains:
            raise ValueError(f"dropout {self.name!r} removes no model terms")

        strategy = self.refit
        if strategy == "auto":
            strategy = "full" if predictors else "gains"
        if strategy == "gains" and predictors:
            raise ValueError(
                f"dropout {self.name!r} removes predictors and therefore requires a full refit"
            )
        return ResolvedDropout(
            name=self.name,
            predictors=tuple(
                p for p in prepared.spec.predictor_names if p in predictors
            ),
            gains=tuple(g for g in prepared.spec.gain_names if g in self.remove_gains),
            refit=strategy,
        )


@dataclass(frozen=True)
class ResolvedDropout:
    name: str
    predictors: tuple[str, ...]
    gains: tuple[str, ...]
    refit: Literal["gains", "full"]


@dataclass(frozen=True)
class CVResult:
    r2_per_fold: np.ndarray
    n_iter_per_fold: np.ndarray

    @property
    def r2(self) -> float:
        return float(np.nanmean(self.r2_per_fold))


@dataclass(frozen=True)
class DropoutResult:
    dropout: ResolvedDropout
    reduced_r2_per_fold: np.ndarray
    delta_r2_per_fold: np.ndarray

    @property
    def reduced_r2(self) -> float:
        return float(np.nanmean(self.reduced_r2_per_fold))

    @property
    def delta_r2(self) -> float:
        return float(np.nanmean(self.delta_r2_per_fold))


@dataclass(frozen=True)
class EvaluationResult:
    fit: FittedModel
    train_r2: float
    cv: CVResult
    dropouts: Mapping[str, DropoutResult]

    def to_dict(self) -> dict:
        """Return a serialization-friendly summary, including fitted kernels."""
        return {
            "model": self.fit.spec.name,
            "train_r2": self.train_r2,
            "cv_r2": self.cv.r2,
            "cv_r2_per_fold": self.cv.r2_per_fold.copy(),
            "n_iter": self.fit.n_iter,
            "n_iter_per_fold": self.cv.n_iter_per_fold.copy(),
            "converged": self.fit.converged,
            "intercept": self.fit.intercept,
            "kernels": self.fit.kernels(),
            "gain_table": self.fit.gain_table(),
            "dropouts": {
                name: {
                    "predictors": result.dropout.predictors,
                    "gains": result.dropout.gains,
                    "refit": result.dropout.refit,
                    "reduced_r2": result.reduced_r2,
                    "reduced_r2_per_fold": result.reduced_r2_per_fold.copy(),
                    "delta_r2": result.delta_r2,
                    "delta_r2_per_fold": result.delta_r2_per_fold.copy(),
                }
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

    requests = tuple(dropouts or ())
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
    full_iterations: list[int] = []
    reduced_r2: dict[str, list[float]] = {dropout.name: [] for dropout in resolved}
    delta_r2: dict[str, list[float]] = {dropout.name: [] for dropout in resolved}

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
        fold_full_r2 = r2_score(values[test_rows], full_prediction)
        full_r2.append(fold_full_r2)
        full_iterations.append(len(full_state.iterations))

        for dropout in resolved:
            keep = gain_keep_mask(
                prepared,
                remove_gains=dropout.gains,
                remove_predictors=dropout.predictors,
            )
            if dropout.refit == "gains":
                reduced_gain, _ = refit_gains(
                    prepared,
                    values[train_rows],
                    train_blocks,
                    train_gains,
                    full_state.beta,
                    full_state.intercept,
                    fit_config,
                    keep=keep,
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
            fold_reduced_r2 = r2_score(values[test_rows], reduced_prediction)
            reduced_r2[dropout.name].append(fold_reduced_r2)
            delta_r2[dropout.name].append(fold_full_r2 - fold_reduced_r2)

    dropout_results = {
        dropout.name: DropoutResult(
            dropout,
            np.asarray(reduced_r2[dropout.name], dtype=float),
            np.asarray(delta_r2[dropout.name], dtype=float),
        )
        for dropout in resolved
    }
    return EvaluationResult(
        fit=fitted,
        train_r2=train_r2,
        cv=CVResult(
            np.asarray(full_r2, dtype=float),
            np.asarray(full_iterations, dtype=int),
        ),
        dropouts=dropout_results,
    )

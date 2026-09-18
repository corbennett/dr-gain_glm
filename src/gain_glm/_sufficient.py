"""Sufficient-statistics implementation of the Ridge ALS solver.

The base predictor blocks are read once to form grouped Gram matrices.  Every
subsequent kernel fit, gain fit, score, and convergence check is evaluated from
those small matrices instead of rebuilding time-bin-level designs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .design import PreparedDesign
from .model import FitConfig, FitState, Iteration

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class _GroupStats:
    gram: np.ndarray
    target_projection: np.ndarray
    column_sum: np.ndarray
    target_sum: float
    target_square_sum: float
    n_rows: int
    gain_values: np.ndarray
    inner_fold: int | None


def _initial_gain(prepared: PreparedDesign) -> np.ndarray:
    gain = np.zeros(prepared.layout.gain_size)
    for index in prepared.layout.gain_offsets.values():
        gain[index] = 1.0
    return gain


def _normalize_kernels(
    prepared: PreparedDesign, beta: np.ndarray, gain: np.ndarray
) -> None:
    for predictor in prepared.spec.predictors:
        if not predictor.gains:
            continue
        sl = prepared.layout.beta_slices[predictor.name]
        kernel = prepared.bases[predictor.name] @ beta[sl]
        norm = float(np.linalg.norm(kernel))
        if norm < 1e-12:
            continue
        beta[sl] /= norm
        gain[prepared.layout.gain_offsets[predictor.name]] *= norm
        for gain_name in predictor.gains:
            index = prepared.layout.gain_coefficients[(gain_name, predictor.name)]
            gain[index] *= norm


def _solve_ridge(system: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    penalized = system.copy()
    penalized.flat[:: penalized.shape[0] + 1] += alpha
    # Cross-products are symmetric analytically.  Remove tiny asymmetries from
    # floating-point accumulation before the positive-definite Ridge solve.
    penalized = (penalized + penalized.T) * 0.5
    return np.linalg.solve(penalized, target)


class _RidgeStatsProblem:
    def __init__(
        self,
        prepared: PreparedDesign,
        y: np.ndarray,
        blocks: Mapping[str, np.ndarray],
        gain_by_time: Mapping[str, np.ndarray],
        trial_index: np.ndarray,
        config: FitConfig,
        retained_gains: np.ndarray,
    ) -> None:
        self.prepared = prepared
        self.config = config
        self.retained_gains = retained_gains
        self.gain_names = tuple(gain.name for gain in prepared.spec.gains)

        n_rows = y.size
        base = np.empty((n_rows, prepared.layout.beta_size), dtype=float)
        for predictor in prepared.spec.predictors:
            base[:, prepared.layout.beta_slices[predictor.name]] = blocks[
                predictor.name
            ]

        needs_inner_cv = config.kernel_alpha is None or (
            config.gain_alpha is None
            and prepared.layout.gain_size > 0
            and retained_gains.any()
        )
        inner_fold = (
            self._inner_fold_by_row(trial_index, config.inner_cv_folds)
            if needs_inner_cv
            else None
        )

        key_columns = [
            np.asarray(gain_by_time[name], dtype=float).ravel()
            for name in self.gain_names
        ]
        if key_columns or inner_fold is not None:
            # Gains and inner-CV assignments are constant within a trial.  Find
            # distinct combinations on the much shorter trial table rather
            # than sorting one key for every time bin.
            _, first_row, trial_inverse = np.unique(
                trial_index, return_index=True, return_inverse=True
            )
            trial_keys = []
            for name, values in zip(self.gain_names, key_columns):
                if values.size != n_rows:
                    raise ValueError(
                        f"gain {name!r} has length {values.size}, "
                        f"expected {n_rows}"
                    )
                trial_values = values[first_row]
                if not np.array_equal(values, trial_values[trial_inverse]):
                    raise ValueError(
                        f"gain {name!r} must be constant within each trial"
                    )
                trial_keys.append(trial_values)
            if inner_fold is not None:
                trial_keys.append(inner_fold[first_row])
            keys = np.column_stack(trial_keys)
            _, trial_group = np.unique(keys, axis=0, return_inverse=True)
            group_index = trial_group[trial_inverse]
        else:
            group_index = np.zeros(n_rows, dtype=int)

        groups: list[_GroupStats] = []
        for group in range(int(group_index.max()) + 1):
            rows = group_index == group
            group_base = base[rows]
            group_target = y[rows]
            first = int(np.flatnonzero(rows)[0])
            values = np.asarray(
                [gain_by_time[name][first] for name in self.gain_names],
                dtype=float,
            )
            groups.append(
                _GroupStats(
                    gram=group_base.T @ group_base,
                    target_projection=group_base.T @ group_target,
                    column_sum=group_base.sum(axis=0),
                    target_sum=float(group_target.sum()),
                    target_square_sum=float(group_target @ group_target),
                    n_rows=int(rows.sum()),
                    gain_values=values,
                    inner_fold=(
                        None if inner_fold is None else int(inner_fold[first])
                    ),
                )
            )

        self.cv_groups = tuple(groups)
        self.groups = (
            self._combine_inner_folds(self.cv_groups)
            if inner_fold is not None
            else self.cv_groups
        )
        self.n_rows = sum(group.n_rows for group in self.groups)
        self.target_sum = sum(group.target_sum for group in self.groups)
        self.target_square_sum = sum(
            group.target_square_sum for group in self.groups
        )
        mean = self.target_sum / self.n_rows
        variance = max(self.target_square_sum / self.n_rows - mean**2, 0.0)
        self.target_scale = max(float(np.sqrt(variance)), np.finfo(float).eps)
        self.inner_folds = tuple(
            sorted(
                {
                    group.inner_fold
                    for group in self.cv_groups
                    if group.inner_fold is not None
                }
            )
        )

    @staticmethod
    def _combine_inner_folds(
        groups: tuple[_GroupStats, ...],
    ) -> tuple[_GroupStats, ...]:
        """Merge fold-specific moments for ordinary full-data ALS updates."""
        by_gain: dict[tuple[float, ...], list[_GroupStats]] = {}
        for group in groups:
            key = tuple(float(value) for value in group.gain_values)
            by_gain.setdefault(key, []).append(group)

        combined = []
        for values, matching in by_gain.items():
            combined.append(
                _GroupStats(
                    gram=sum(
                        (group.gram for group in matching),
                        np.zeros_like(matching[0].gram),
                    ),
                    target_projection=sum(
                        (group.target_projection for group in matching),
                        np.zeros_like(matching[0].target_projection),
                    ),
                    column_sum=sum(
                        (group.column_sum for group in matching),
                        np.zeros_like(matching[0].column_sum),
                    ),
                    target_sum=sum(group.target_sum for group in matching),
                    target_square_sum=sum(
                        group.target_square_sum for group in matching
                    ),
                    n_rows=sum(group.n_rows for group in matching),
                    gain_values=np.asarray(values),
                    inner_fold=None,
                )
            )
        return tuple(combined)

    @staticmethod
    def _inner_fold_by_row(
        trial_index: np.ndarray, requested_folds: int | None
    ) -> np.ndarray:
        # Import lazily to avoid a module cycle while _solver imports this
        # implementation from its public fit_state wrapper.
        from ._solver import _trial_cv

        splitter = _trial_cv(trial_index, requested_folds)
        fold_by_row = np.full(trial_index.size, -1, dtype=int)
        for fold, (_, validation_rows) in enumerate(
            splitter.split(np.zeros(trial_index.size))
        ):
            fold_by_row[validation_rows] = fold
        if np.any(fold_by_row < 0):
            raise RuntimeError("inner CV did not assign every row to a fold")
        return fold_by_row

    def _groups(self, *, validation_fold: int | None, train: bool):
        if validation_fold is None:
            return self.groups
        return tuple(
            group
            for group in self.cv_groups
            if (group.inner_fold != validation_fold) == train
        )

    def _kernel_scale(self, gain: np.ndarray, group: _GroupStats) -> np.ndarray:
        scale = np.ones(self.prepared.layout.beta_size)
        gain_value = dict(zip(self.gain_names, group.gain_values))
        for predictor in self.prepared.spec.predictors:
            if not predictor.gains:
                continue
            weight = gain[self.prepared.layout.gain_offsets[predictor.name]]
            for name in predictor.gains:
                index = self.prepared.layout.gain_coefficients[
                    (name, predictor.name)
                ]
                weight += gain[index] * gain_value[name]
            scale[self.prepared.layout.beta_slices[predictor.name]] = weight
        return scale

    def _kernel_fit_for_groups(
        self,
        gain: np.ndarray,
        alpha: float,
        groups: tuple[_GroupStats, ...],
    ) -> tuple[np.ndarray, float]:
        size = self.prepared.layout.beta_size
        system = np.zeros((size, size))
        target = np.zeros(size)
        column_sum = np.zeros(size)
        target_sum = 0.0
        n_rows = 0
        for group in groups:
            scale = self._kernel_scale(gain, group)
            system += scale[:, None] * group.gram * scale[None, :]
            target += scale * group.target_projection
            column_sum += scale * group.column_sum
            target_sum += group.target_sum
            n_rows += group.n_rows

        target_mean = target_sum / n_rows
        system -= np.outer(column_sum, column_sum) / n_rows
        target -= column_sum * target_mean
        beta = _solve_ridge(system, target, alpha)
        intercept = target_mean - (column_sum @ beta) / n_rows
        return beta, float(intercept)

    def _kernel_sse(
        self,
        gain: np.ndarray,
        beta: np.ndarray,
        intercept: float,
        groups: tuple[_GroupStats, ...],
    ) -> float:
        coefficients = [
            self._kernel_scale(gain, group) * beta for group in groups
        ]
        return self._prediction_sse(intercept, coefficients, groups)

    def fit_kernel(
        self, gain: np.ndarray, alpha: float | None
    ) -> tuple[np.ndarray, float, float]:
        selected = alpha
        if selected is None:
            mean_mse = []
            for candidate in self.config.alphas:
                fold_mse = []
                for fold in self.inner_folds:
                    train = self._groups(validation_fold=fold, train=True)
                    validation = self._groups(validation_fold=fold, train=False)
                    beta, intercept = self._kernel_fit_for_groups(
                        gain, candidate, train
                    )
                    fold_mse.append(
                        self._kernel_sse(
                            gain, beta, intercept, validation
                        )
                        / sum(group.n_rows for group in validation)
                    )
                mean_mse.append(float(np.mean(fold_mse)))
            selected = float(self.config.alphas[int(np.argmin(mean_mse))])
        beta, intercept = self._kernel_fit_for_groups(
            gain, selected, self.groups
        )
        return beta, intercept, selected

    def _gain_map(
        self, beta: np.ndarray, group: _GroupStats
    ) -> tuple[np.ndarray, np.ndarray]:
        mapping = np.zeros(
            (self.prepared.layout.beta_size, self.prepared.layout.gain_size)
        )
        fixed = np.zeros(self.prepared.layout.beta_size)
        gain_value = dict(zip(self.gain_names, group.gain_values))
        for predictor in self.prepared.spec.predictors:
            sl = self.prepared.layout.beta_slices[predictor.name]
            if not predictor.gains:
                fixed[sl] = beta[sl]
                continue
            mapping[
                sl, self.prepared.layout.gain_offsets[predictor.name]
            ] = beta[sl]
            for name in predictor.gains:
                index = self.prepared.layout.gain_coefficients[
                    (name, predictor.name)
                ]
                mapping[sl, index] = beta[sl] * gain_value[name]
        return mapping, fixed

    def _gain_group_terms(
        self, beta: np.ndarray, intercept: float, group: _GroupStats
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        mapping, fixed = self._gain_map(beta, group)
        residual_projection = (
            group.target_projection
            - group.gram @ fixed
            - intercept * group.column_sum
        )
        system = mapping.T @ group.gram @ mapping
        target = mapping.T @ residual_projection
        residual_square_sum = (
            group.target_square_sum
            - 2 * intercept * group.target_sum
            - 2 * fixed @ group.target_projection
            + group.n_rows * intercept**2
            + 2 * intercept * (group.column_sum @ fixed)
            + fixed @ group.gram @ fixed
        )
        return system, target, float(residual_square_sum), fixed

    def _gain_fit_for_groups(
        self,
        beta: np.ndarray,
        intercept: float,
        alpha: float,
        groups: tuple[_GroupStats, ...],
    ) -> np.ndarray:
        gain = np.zeros(self.prepared.layout.gain_size)
        if not self.retained_gains.any():
            return gain
        system = np.zeros(
            (self.prepared.layout.gain_size, self.prepared.layout.gain_size)
        )
        target = np.zeros(self.prepared.layout.gain_size)
        for group in groups:
            group_system, group_target, _, _ = self._gain_group_terms(
                beta, intercept, group
            )
            system += group_system
            target += group_target
        keep = self.retained_gains
        gain[keep] = _solve_ridge(
            system[np.ix_(keep, keep)], target[keep], alpha
        )
        return gain

    def _gain_sse(
        self,
        beta: np.ndarray,
        intercept: float,
        gain: np.ndarray,
        groups: tuple[_GroupStats, ...],
    ) -> float:
        result = 0.0
        keep = self.retained_gains
        retained_gain = gain[keep]
        for group in groups:
            system, target, residual_square_sum, _ = self._gain_group_terms(
                beta, intercept, group
            )
            result += residual_square_sum
            if retained_gain.size:
                retained_system = system[np.ix_(keep, keep)]
                result += (
                    -2 * retained_gain @ target[keep]
                    + retained_gain @ retained_system @ retained_gain
                )
        return float(max(result, 0.0))

    def fit_gain(
        self, beta: np.ndarray, intercept: float, alpha: float | None
    ) -> tuple[np.ndarray, float | None]:
        if self.prepared.layout.gain_size == 0 or not self.retained_gains.any():
            return np.zeros(self.prepared.layout.gain_size), None
        selected = alpha
        if selected is None:
            mean_mse = []
            for candidate in self.config.alphas:
                fold_mse = []
                for fold in self.inner_folds:
                    train = self._groups(validation_fold=fold, train=True)
                    validation = self._groups(validation_fold=fold, train=False)
                    gain = self._gain_fit_for_groups(
                        beta, intercept, candidate, train
                    )
                    fold_mse.append(
                        self._gain_sse(beta, intercept, gain, validation)
                        / sum(group.n_rows for group in validation)
                    )
                mean_mse.append(float(np.mean(fold_mse)))
            selected = float(self.config.alphas[int(np.argmin(mean_mse))])
        return (
            self._gain_fit_for_groups(beta, intercept, selected, self.groups),
            selected,
        )

    def prediction_coefficients(
        self, beta: np.ndarray, gain: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        coefficients = []
        for group in self.groups:
            mapping, fixed = self._gain_map(beta, group)
            coefficients.append(fixed + mapping @ gain)
        return tuple(coefficients)

    @staticmethod
    def _prediction_sse(
        intercept: float,
        coefficients: list[np.ndarray] | tuple[np.ndarray, ...],
        groups: tuple[_GroupStats, ...],
    ) -> float:
        result = 0.0
        for group, coefficient in zip(groups, coefficients):
            result += (
                group.target_square_sum
                - 2 * intercept * group.target_sum
                - 2 * coefficient @ group.target_projection
                + group.n_rows * intercept**2
                + 2 * intercept * (group.column_sum @ coefficient)
                + coefficient @ group.gram @ coefficient
            )
        return float(max(result, 0.0))

    def prediction_sse(
        self, intercept: float, coefficients: tuple[np.ndarray, ...]
    ) -> float:
        return self._prediction_sse(intercept, coefficients, self.groups)

    def prediction_difference(
        self,
        intercept: float,
        coefficients: tuple[np.ndarray, ...],
        previous_intercept: float,
        previous_coefficients: tuple[np.ndarray, ...],
    ) -> float:
        intercept_change = intercept - previous_intercept
        result = 0.0
        for group, coefficient, previous in zip(
            self.groups, coefficients, previous_coefficients
        ):
            change = coefficient - previous
            result += (
                group.n_rows * intercept_change**2
                + 2 * intercept_change * (group.column_sum @ change)
                + change @ group.gram @ change
            )
        return float(max(result, 0.0))


def fit_state_sufficient(
    prepared: PreparedDesign,
    y: np.ndarray,
    blocks: Mapping[str, np.ndarray],
    gain_by_time: Mapping[str, np.ndarray],
    config: FitConfig,
    *,
    trial_index: np.ndarray,
    keep_gains: np.ndarray | None = None,
) -> FitState:
    """Run Ridge ALS using grouped cross-products instead of row designs."""
    trial_ids = np.asarray(trial_index).ravel()
    if trial_ids.size != y.size:
        raise ValueError(
            f"trial_index has length {trial_ids.size}, expected {y.size}"
        )
    retained = (
        np.ones(prepared.layout.gain_size, dtype=bool)
        if keep_gains is None
        else np.asarray(keep_gains, dtype=bool).ravel()
    )
    if retained.size != prepared.layout.gain_size:
        raise ValueError("gain keep mask has the wrong length")

    problem = _RidgeStatsProblem(
        prepared,
        np.asarray(y, dtype=float).ravel(),
        blocks,
        gain_by_time,
        trial_ids,
        config,
        retained,
    )
    gain = _initial_gain(prepared)
    gain[~retained] = 0
    beta = np.zeros(prepared.layout.beta_size)
    intercept = 0.0
    kernel_alpha = config.kernel_alpha
    gain_alpha = config.gain_alpha
    variable_indices = [
        index
        for index in prepared.layout.gain_coefficients.values()
        if retained[index]
    ]
    previous = gain[variable_indices].copy()
    previous_mse = problem.target_square_sum / problem.n_rows
    previous_intercept = 0.0
    previous_coefficients = tuple(
        np.zeros(prepared.layout.beta_size) for _ in problem.groups
    )
    stable = 0
    converged = False
    iterations: list[Iteration] = []

    for iteration in range(config.max_iter):
        beta, intercept, selected_kernel_alpha = problem.fit_kernel(
            gain, kernel_alpha
        )
        if kernel_alpha is None:
            kernel_alpha = selected_kernel_alpha
        _normalize_kernels(prepared, beta, gain)
        gain, selected_gain_alpha = problem.fit_gain(
            beta, intercept, gain_alpha
        )
        if gain_alpha is None:
            gain_alpha = selected_gain_alpha

        coefficients = problem.prediction_coefficients(beta, gain)
        mse = problem.prediction_sse(intercept, coefficients) / problem.n_rows
        relative_mse_change = abs(mse - previous_mse) / max(
            abs(previous_mse), np.finfo(float).eps
        )
        relative_prediction_change = float(
            np.sqrt(
                problem.prediction_difference(
                    intercept,
                    coefficients,
                    previous_intercept,
                    previous_coefficients,
                )
                / problem.n_rows
            )
            / problem.target_scale
        )
        if variable_indices:
            current = gain[variable_indices]
            absolute_gain_change = np.abs(current - previous)
            max_abs_gain_change = float(np.max(absolute_gain_change))
            relative_gain_change = float(
                np.max(absolute_gain_change / (1.0 + np.abs(previous)))
            )
        else:
            current = np.zeros(0)
            max_abs_gain_change = 0.0
            relative_gain_change = 0.0
        iterations.append(
            Iteration(
                iteration,
                mse,
                float(kernel_alpha),
                gain_alpha,
                relative_mse_change,
                relative_prediction_change,
                relative_gain_change,
                max_abs_gain_change,
            )
        )
        if config.verbose:
            print(
                f"iter {iteration:3d} mse={mse:.6g} "
                f"kernel_alpha={kernel_alpha:.3g} gain_alpha={gain_alpha} "
                f"relative_mse_change={relative_mse_change:.3g} "
                f"relative_prediction_change={relative_prediction_change:.3g} "
                f"max_abs_gain_change={max_abs_gain_change:.3g}"
            )

        if not variable_indices:
            converged = True
            break
        if max_abs_gain_change <= config.tol:
            stable += 1
            if stable >= config.patience:
                converged = True
                break
        else:
            stable = 0
        previous = current.copy()
        previous_mse = mse
        previous_intercept = intercept
        previous_coefficients = tuple(
            coefficient.copy() for coefficient in coefficients
        )

    gain, final_gain_alpha = problem.fit_gain(beta, intercept, gain_alpha)
    if iterations:
        coefficients = problem.prediction_coefficients(beta, gain)
        final = iterations[-1]
        iterations[-1] = Iteration(
            final.number,
            problem.prediction_sse(intercept, coefficients) / problem.n_rows,
            final.kernel_alpha,
            final_gain_alpha,
            final.relative_mse_change,
            final.relative_prediction_change,
            final.relative_gain_change,
            final.max_abs_gain_change,
        )
    return FitState(beta, gain, intercept, tuple(iterations), converged)

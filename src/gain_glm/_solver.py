"""Numerical implementation of the bilinear alternating fit.

This module is private on purpose. Public model objects are immutable; every
fit returns a new :class:`gain_glm.model.FittedModel`.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV

from .design import PreparedDesign
from .model import FitConfig, FitState, FittedModel, Iteration


def r2_score(y: np.ndarray, prediction: np.ndarray) -> float:
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0:
        return float("nan")
    return 1 - float(np.sum((y - prediction) ** 2)) / total


def _initial_gain(prepared: PreparedDesign) -> np.ndarray:
    gain = np.zeros(prepared.layout.gain_size)
    for index in prepared.layout.gain_offsets.values():
        gain[index] = 1.0
    return gain


def _kernel_design(
    prepared: PreparedDesign,
    blocks: Mapping[str, np.ndarray],
    gain_by_time: Mapping[str, np.ndarray],
    gain: np.ndarray,
) -> np.ndarray:
    n_time = next(iter(blocks.values())).shape[0]
    design = np.zeros((n_time, prepared.layout.beta_size))
    for predictor in prepared.spec.predictors:
        block = blocks[predictor.name]
        sl = prepared.layout.beta_slices[predictor.name]
        if not predictor.gains:
            design[:, sl] = block
            continue
        weight = np.full(
            n_time, gain[prepared.layout.gain_offsets[predictor.name]]
        )
        for gain_name in predictor.gains:
            index = prepared.layout.gain_coefficients[(gain_name, predictor.name)]
            weight += gain[index] * gain_by_time[gain_name]
        design[:, sl] = block * weight[:, None]
    return design


def _kernel_drives(
    prepared: PreparedDesign,
    blocks: Mapping[str, np.ndarray],
    beta: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        predictor.name: blocks[predictor.name]
        @ beta[prepared.layout.beta_slices[predictor.name]]
        for predictor in prepared.spec.predictors
    }


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
            gain[
                prepared.layout.gain_coefficients[(gain_name, predictor.name)]
            ] *= norm


def _gain_design(
    prepared: PreparedDesign,
    drives: Mapping[str, np.ndarray],
    gain_by_time: Mapping[str, np.ndarray],
) -> np.ndarray | None:
    if prepared.layout.gain_size == 0:
        return None
    n_time = next(iter(drives.values())).size
    design = np.zeros((n_time, prepared.layout.gain_size))
    for predictor in prepared.spec.predictors:
        if not predictor.gains:
            continue
        design[:, prepared.layout.gain_offsets[predictor.name]] = drives[
            predictor.name
        ]
        for gain_name in predictor.gains:
            index = prepared.layout.gain_coefficients[(gain_name, predictor.name)]
            design[:, index] = gain_by_time[gain_name] * drives[predictor.name]
    return design


def gain_keep_mask(
    prepared: PreparedDesign,
    *,
    remove_gains: Sequence[str] = (),
    remove_predictors: Sequence[str] = (),
) -> np.ndarray:
    """Return gain-vector columns retained by a reduced model."""
    unknown_gains = set(remove_gains) - set(prepared.spec.gain_names)
    unknown_predictors = set(remove_predictors) - set(prepared.spec.predictor_names)
    if unknown_gains:
        raise ValueError(f"unknown gains: {sorted(unknown_gains)}")
    if unknown_predictors:
        raise ValueError(f"unknown predictors: {sorted(unknown_predictors)}")

    keep = np.ones(prepared.layout.gain_size, dtype=bool)
    for (gain_name, predictor), index in prepared.layout.gain_coefficients.items():
        if gain_name in remove_gains or predictor in remove_predictors:
            keep[index] = False
    for predictor in remove_predictors:
        if predictor in prepared.layout.gain_offsets:
            keep[prepared.layout.gain_offsets[predictor]] = False
    return keep


def _fit_gain_coefficients(
    prepared: PreparedDesign,
    config: FitConfig,
    design: np.ndarray | None,
    target: np.ndarray,
    *,
    keep: np.ndarray | None = None,
    alpha: float | None = None,
) -> tuple[np.ndarray, float | None]:
    gain = np.zeros(prepared.layout.gain_size)
    if design is None or prepared.layout.gain_size == 0:
        return gain, None
    retained = (
        np.ones(prepared.layout.gain_size, dtype=bool)
        if keep is None
        else np.asarray(keep, dtype=bool).ravel()
    )
    if retained.size != prepared.layout.gain_size:
        raise ValueError("gain keep mask has the wrong length")
    if not retained.any():
        return gain, None

    x = design[:, retained]
    if alpha is None:
        solver = RidgeCV(
            alphas=np.asarray(config.alphas),
            cv=config.inner_cv_folds,
            fit_intercept=False,
        )
        solver.fit(x, target)
        selected = float(solver.alpha_)
    else:
        solver = Ridge(alpha=alpha, fit_intercept=False)
        solver.fit(x, target)
        selected = float(alpha)
    gain[retained] = solver.coef_
    return gain, selected


def refit_gains(
    prepared: PreparedDesign,
    y: np.ndarray,
    blocks: Mapping[str, np.ndarray],
    gain_by_time: Mapping[str, np.ndarray],
    beta: np.ndarray,
    intercept: float,
    config: FitConfig,
    *,
    keep: np.ndarray | None = None,
) -> tuple[np.ndarray, float | None]:
    """Fit gain parameters while holding kernels and intercept fixed."""
    drives = _kernel_drives(prepared, blocks, beta)
    offset = np.full(y.size, intercept)
    for predictor in prepared.spec.predictors:
        if not predictor.gains:
            offset += drives[predictor.name]
    return _fit_gain_coefficients(
        prepared,
        config,
        _gain_design(prepared, drives, gain_by_time),
        y - offset,
        keep=keep,
        alpha=config.gain_alpha,
    )


def predict_parameters(
    prepared: PreparedDesign,
    blocks: Mapping[str, np.ndarray],
    gain_by_time: Mapping[str, np.ndarray],
    beta: np.ndarray,
    gain: np.ndarray,
    intercept: float,
) -> np.ndarray:
    drives = _kernel_drives(prepared, blocks, beta)
    prediction = np.full(next(iter(drives.values())).size, intercept)
    for predictor in prepared.spec.predictors:
        drive = drives[predictor.name]
        if not predictor.gains:
            prediction += drive
            continue
        weight = np.full(
            drive.size, gain[prepared.layout.gain_offsets[predictor.name]]
        )
        for gain_name in predictor.gains:
            index = prepared.layout.gain_coefficients[(gain_name, predictor.name)]
            weight += gain[index] * gain_by_time[gain_name]
        prediction += weight * drive
    return prediction


def predict_state(
    prepared: PreparedDesign,
    blocks: Mapping[str, np.ndarray],
    state: FitState,
) -> np.ndarray:
    return predict_parameters(
        prepared,
        blocks,
        prepared.gain_by_time,
        state.beta,
        state.gain,
        state.intercept,
    )


def fit_state(
    prepared: PreparedDesign,
    y: np.ndarray,
    blocks: Mapping[str, np.ndarray],
    gain_by_time: Mapping[str, np.ndarray],
    config: FitConfig,
    *,
    keep_gains: np.ndarray | None = None,
) -> FitState:
    """Run ALS on arrays already restricted to the desired training rows."""
    gain = _initial_gain(prepared)
    beta = np.zeros(prepared.layout.beta_size)
    intercept = 0.0
    retained = (
        np.ones(prepared.layout.gain_size, dtype=bool)
        if keep_gains is None
        else np.asarray(keep_gains, dtype=bool).ravel()
    )
    if retained.size != prepared.layout.gain_size:
        raise ValueError("gain keep mask has the wrong length")
    gain[~retained] = 0

    kernel_alpha = config.kernel_alpha
    gain_alpha = config.gain_alpha
    kernel_solver = None
    variable_indices = [
        index
        for index in prepared.layout.gain_coefficients.values()
        if retained[index]
    ]
    previous = gain[variable_indices].copy()
    stable = 0
    iterations: list[Iteration] = []

    for iteration in range(config.max_iter):
        x = _kernel_design(prepared, blocks, gain_by_time, gain)
        if config.regularizer == "lasso":
            if kernel_alpha is None:
                solver = LassoCV(
                    alphas=np.asarray(config.alphas),
                    cv=config.inner_cv_folds,
                    fit_intercept=True,
                    max_iter=10_000,
                )
                solver.fit(x, y)
                kernel_alpha = float(solver.alpha_)
                beta = solver.coef_.copy()
                intercept = float(solver.intercept_)
            else:
                if kernel_solver is None:
                    kernel_solver = Lasso(
                        alpha=kernel_alpha,
                        fit_intercept=True,
                        max_iter=10_000,
                        warm_start=True,
                    )
                    kernel_solver.coef_ = beta.copy()
                    kernel_solver.intercept_ = intercept
                kernel_solver.fit(x, y)
                beta = kernel_solver.coef_.copy()
                intercept = float(kernel_solver.intercept_)
        else:
            if kernel_alpha is None:
                solver = RidgeCV(
                    alphas=np.asarray(config.alphas),
                    cv=config.inner_cv_folds,
                    fit_intercept=True,
                )
                solver.fit(x, y)
                kernel_alpha = float(solver.alpha_)
                beta = solver.coef_.copy()
                intercept = float(solver.intercept_)
            else:
                if kernel_solver is None:
                    kernel_solver = Ridge(alpha=kernel_alpha, fit_intercept=True)
                kernel_solver.fit(x, y)
                beta = kernel_solver.coef_.copy()
                intercept = float(kernel_solver.intercept_)

        _normalize_kernels(prepared, beta, gain)
        if kernel_solver is not None and hasattr(kernel_solver, "coef_"):
            kernel_solver.coef_ = beta.copy()

        drives = _kernel_drives(prepared, blocks, beta)
        offset = np.full(y.size, intercept)
        for predictor in prepared.spec.predictors:
            if not predictor.gains:
                offset += drives[predictor.name]
        gain, selected_gain_alpha = _fit_gain_coefficients(
            prepared,
            config,
            _gain_design(prepared, drives, gain_by_time),
            y - offset,
            keep=retained,
            alpha=gain_alpha,
        )
        if gain_alpha is None:
            gain_alpha = selected_gain_alpha

        prediction = predict_parameters(
            prepared, blocks, gain_by_time, beta, gain, intercept
        )
        mse = float(np.mean((y - prediction) ** 2))
        iterations.append(
            Iteration(iteration, mse, float(kernel_alpha), gain_alpha)
        )
        if config.verbose:
            print(
                f"iter {iteration:3d} mse={mse:.6g} "
                f"kernel_alpha={kernel_alpha:.3g} gain_alpha={gain_alpha}"
            )

        if not variable_indices:
            break
        current = gain[variable_indices]
        if np.max(np.abs(current - previous)) <= config.tol:
            stable += 1
            if stable >= config.patience:
                break
        else:
            stable = 0
        previous = current.copy()

    gain, final_gain_alpha = refit_gains(
        prepared,
        y,
        blocks,
        gain_by_time,
        beta,
        intercept,
        config,
        keep=retained,
    )
    if iterations:
        prediction = predict_parameters(
            prepared, blocks, gain_by_time, beta, gain, intercept
        )
        final = iterations[-1]
        iterations[-1] = Iteration(
            final.number,
            float(np.mean((y - prediction) ** 2)),
            final.kernel_alpha,
            final_gain_alpha,
        )
    return FitState(beta, gain, intercept, tuple(iterations))


def _fit_mask(prepared: PreparedDesign, mask: np.ndarray | None) -> np.ndarray:
    used = prepared.fit_mask if mask is None else np.asarray(mask, dtype=bool).ravel()
    if used.size != prepared.data.n_time:
        raise ValueError(f"mask has length {used.size}, expected {prepared.data.n_time}")
    if not used.any():
        raise ValueError("mask selects no bins")
    return used


def fit_model(
    prepared: PreparedDesign,
    y: np.ndarray,
    *,
    config: FitConfig | None = None,
    mask: np.ndarray | None = None,
) -> FittedModel:
    """Fit a prepared design and return immutable fitted parameters."""
    values = np.asarray(y, dtype=float).ravel()
    if values.size != prepared.data.n_time:
        raise ValueError(
            f"target has length {values.size}, expected {prepared.data.n_time}"
        )
    settings = FitConfig() if config is None else config
    used = _fit_mask(prepared, mask)
    blocks = prepared.blocks_for_target(values if prepared.has_history else None)
    state = fit_state(
        prepared,
        values[used],
        {name: block[used] for name, block in blocks.items()},
        {name: gain[used] for name, gain in prepared.gain_by_time.items()},
        settings,
    )
    return FittedModel(prepared, state, settings, used.copy())

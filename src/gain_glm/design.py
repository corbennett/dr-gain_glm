"""Compile declarative model specifications into reusable numerical designs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .data import ModelData, TimedSignal
from .model import Event, FitConfig, History, ModelSpec, Signal


def linear_cosine_basis(n_basis: int, n_lag_bins: int) -> np.ndarray:
    if n_basis < 1:
        raise ValueError("n_basis must be positive")
    if n_basis == 1:
        return np.ones((n_lag_bins, 1))
    centers = np.linspace(0, n_lag_bins - 1, n_basis)
    spacing = centers[1] - centers[0]
    width = 2 * spacing
    time = np.arange(n_lag_bins, dtype=float)
    basis = np.zeros((n_lag_bins, n_basis))
    for column, center in enumerate(centers):
        phase = (time - center) * np.pi / width
        bump = (np.cos(np.clip(phase, -np.pi, np.pi)) + 1) / 2
        bump[np.abs(phase) > np.pi] = 0
        basis[:, column] = bump
    return basis


def log_cosine_basis(
    n_basis: int, n_lag_bins: int, log_offset: float = 1.0
) -> np.ndarray:
    if n_basis < 1:
        raise ValueError("n_basis must be positive")
    if n_basis == 1:
        return np.ones((n_lag_bins, 1))
    nonlinear_time = np.log(np.arange(n_lag_bins, dtype=float) + log_offset)
    centers = np.linspace(nonlinear_time[0], nonlinear_time[-1], n_basis)
    spacing = centers[1] - centers[0]
    width = 2 * spacing
    basis = np.zeros((n_lag_bins, n_basis))
    for column, center in enumerate(centers):
        phase = (nonlinear_time - center) * np.pi / width
        bump = (np.cos(np.clip(phase, -np.pi, np.pi)) + 1) / 2
        bump[np.abs(phase) > np.pi] = 0
        basis[:, column] = bump
    return basis


def make_basis(name: str, n_basis: int, n_lag_bins: int) -> np.ndarray:
    if name == "cosine":
        return linear_cosine_basis(n_basis, n_lag_bins)
    if name == "log_cosine":
        return log_cosine_basis(n_basis, n_lag_bins)
    if name == "identity":
        if n_basis != n_lag_bins:
            raise ValueError(
                "identity basis requires n_basis to equal the number of lag bins"
            )
        return np.eye(n_lag_bins)
    raise ValueError(f"unknown basis {name!r}")


def _lags(window: tuple[float, float], dt: float) -> np.ndarray:
    low = int(np.floor(window[0] / dt))
    high = int(np.ceil(window[1] / dt))
    return np.arange(low, high + 1)


def _event_series(times: np.ndarray, dt: float, n_time: int) -> np.ndarray:
    series = np.zeros(n_time)
    bins = np.floor(np.asarray(times, dtype=float).ravel() / dt).astype(int)
    bins = bins[(bins >= 0) & (bins < n_time)]
    np.add.at(series, bins, 1.0)
    return series


def _design_block(
    series: np.ndarray, lags: np.ndarray, basis: np.ndarray
) -> np.ndarray:
    n_time = series.size
    lagged = np.zeros((n_time, lags.size))
    for column, lag in enumerate(lags):
        if lag >= 0 and lag < n_time:
            lagged[lag:, column] = series[: n_time - lag]
        elif lag < 0 and -lag < n_time:
            lagged[: n_time + lag, column] = series[-lag:]
    return lagged @ basis


def _mark_outliers(values: np.ndarray, threshold: float | None) -> np.ndarray:
    if threshold is None or threshold <= 0:
        return values.astype(float, copy=True)
    output = values.astype(float, copy=True)
    for _ in range(5):
        finite = np.isfinite(output)
        if finite.sum() < 2:
            break
        sample = output[finite]
        median = float(np.median(sample))
        mad = float(np.median(np.abs(sample - median)))
        scale = 1.4826 * mad if mad >= 1e-12 else float(np.std(sample))
        if scale < 1e-12:
            break
        remove = finite & (np.abs((output - median) / scale) > threshold)
        if not remove.any():
            break
        output[remove] = np.nan
    return output


def _normalize(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return values
    centered = values - float(np.mean(values))
    if mode == "center":
        return centered
    if mode == "zscore":
        scale = float(np.std(values))
        return centered if scale < 1e-12 else centered / scale
    raise ValueError(f"unknown normalization {mode!r}")


def _resample_signal(
    signal: TimedSignal, predictor: Signal, data: ModelData
) -> np.ndarray:
    values = _mark_outliers(signal.values, predictor.outlier_zscore)
    if signal.times is None:
        if values.size != data.n_time:
            raise ValueError(
                f"signal source {predictor.source!r} has length {values.size}; "
                f"expected {data.n_time} or explicit sample times"
            )
        missing = np.isnan(values)
        if missing.all():
            raise ValueError(f"signal source {predictor.source!r} is all NaN")
        if missing.any():
            index = np.arange(data.n_time)
            values = np.where(
                missing,
                np.interp(index, index[~missing], values[~missing]),
                values,
            )
        return _normalize(values, predictor.normalize)

    times = signal.times
    valid = np.isfinite(values) & np.isfinite(times)
    if not valid.any():
        raise ValueError(f"signal source {predictor.source!r} has no finite samples")
    values = values[valid]
    times = times[valid]
    order = np.argsort(times)
    values = values[order]
    times = times[order]

    if predictor.align == "interp":
        centers = (np.arange(data.n_time) + 0.5) * data.dt
        output = np.interp(centers, times, values)
    else:
        bins = np.floor(times / data.dt).astype(int)
        in_range = (bins >= 0) & (bins < data.n_time)
        bins = bins[in_range]
        values = values[in_range]
        sums = np.zeros(data.n_time)
        counts = np.zeros(data.n_time, dtype=int)
        np.add.at(sums, bins, values)
        np.add.at(counts, bins, 1)
        output = np.zeros(data.n_time)
        populated = counts > 0
        output[populated] = sums[populated] / counts[populated]
    return _normalize(output, predictor.normalize)


@dataclass(frozen=True)
class ParameterLayout:
    beta_slices: Mapping[str, slice]
    beta_size: int
    gain_offsets: Mapping[str, int]
    gain_coefficients: Mapping[tuple[str, str], int]
    gain_size: int


@dataclass(frozen=True)
class PreparedDesign:
    """A validated, session-specific design reusable across target units."""

    spec: ModelSpec
    data: ModelData
    base_blocks: Mapping[str, np.ndarray]
    gain_by_time: Mapping[str, np.ndarray]
    lags: Mapping[str, np.ndarray]
    bases: Mapping[str, np.ndarray]
    layout: ParameterLayout
    fit_mask: np.ndarray

    @property
    def has_history(self) -> bool:
        return any(isinstance(p, History) for p in self.spec.predictors)

    @property
    def history_lag_bins(self) -> int:
        history = [p for p in self.spec.predictors if isinstance(p, History)]
        return max((int(self.lags[p.name].max()) for p in history), default=0)

    def blocks_for_target(self, target: np.ndarray | None = None) -> dict[str, np.ndarray]:
        blocks = dict(self.base_blocks)
        history_terms = [p for p in self.spec.predictors if isinstance(p, History)]
        if history_terms:
            if target is None:
                raise ValueError("target values are required by history predictors")
            values = np.asarray(target, dtype=float).ravel()
            if values.size != self.data.n_time:
                raise ValueError(
                    f"target has length {values.size}, expected {self.data.n_time}"
                )
            for predictor in history_terms:
                blocks[predictor.name] = _design_block(
                    values, self.lags[predictor.name], self.bases[predictor.name]
                )
        return blocks

    def fit(
        self,
        y: np.ndarray,
        *,
        config: FitConfig | None = None,
        mask: np.ndarray | None = None,
    ):
        from ._solver import fit_model

        return fit_model(self, y, config=config, mask=mask)

    def evaluate(
        self,
        y: np.ndarray,
        *,
        fit: FitConfig | None = None,
        cv=None,
        dropouts: Sequence | None = None,
        mask: np.ndarray | None = None,
    ):
        from .evaluation import evaluate

        return evaluate(self, y, fit=fit, cv=cv, dropouts=dropouts, mask=mask)


def _parameter_layout(spec: ModelSpec, bases: Mapping[str, np.ndarray]) -> ParameterLayout:
    beta_slices: dict[str, slice] = {}
    beta_start = 0
    for predictor in spec.predictors:
        beta_end = beta_start + bases[predictor.name].shape[1]
        beta_slices[predictor.name] = slice(beta_start, beta_end)
        beta_start = beta_end

    gain_offsets: dict[str, int] = {}
    gain_coefficients: dict[tuple[str, str], int] = {}
    gain_index = 0
    for predictor in spec.predictors:
        if predictor.gains:
            gain_offsets[predictor.name] = gain_index
            gain_index += 1
    for predictor in spec.predictors:
        for gain in predictor.gains:
            gain_coefficients[(gain, predictor.name)] = gain_index
            gain_index += 1
    return ParameterLayout(
        beta_slices=beta_slices,
        beta_size=beta_start,
        gain_offsets=gain_offsets,
        gain_coefficients=gain_coefficients,
        gain_size=gain_index,
    )


def compile_design(
    spec: ModelSpec,
    data: ModelData,
    *,
    fit_mask: np.ndarray | None = None,
) -> PreparedDesign:
    """Resolve all named inputs and precompute target-independent convolutions."""
    if fit_mask is None:
        mask = np.ones(data.n_time, dtype=bool)
    else:
        mask = np.asarray(fit_mask, dtype=bool).ravel()
        if mask.size != data.n_time:
            raise ValueError(f"fit_mask has length {mask.size}, expected {data.n_time}")
        if not mask.any():
            raise ValueError("fit_mask selects no bins")

    lags: dict[str, np.ndarray] = {}
    bases: dict[str, np.ndarray] = {}
    blocks: dict[str, np.ndarray] = {}
    for predictor in spec.predictors:
        predictor_lags = _lags(predictor.window, data.dt)
        if isinstance(predictor, History) and predictor_lags.min() < 1:
            raise ValueError(
                f"history predictor {predictor.name!r} must start at least one bin in the past"
            )
        predictor_basis = make_basis(
            predictor.basis, predictor.n_basis, predictor_lags.size
        )
        lags[predictor.name] = predictor_lags
        bases[predictor.name] = predictor_basis

        if isinstance(predictor, Event):
            if predictor.source not in data.events:
                raise KeyError(
                    f"event source {predictor.source!r} required by {predictor.name!r} is missing"
                )
            series = _event_series(
                data.events[predictor.source], data.dt, data.n_time
            )
            blocks[predictor.name] = _design_block(
                series, predictor_lags, predictor_basis
            )
        elif isinstance(predictor, Signal):
            if predictor.source not in data.signals:
                raise KeyError(
                    f"signal source {predictor.source!r} required by {predictor.name!r} is missing"
                )
            series = _resample_signal(data.signals[predictor.source], predictor, data)
            blocks[predictor.name] = _design_block(
                series, predictor_lags, predictor_basis
            )

    gain_by_time: dict[str, np.ndarray] = {}
    for gain in spec.gains:
        if gain.source not in data.trial_values:
            raise KeyError(
                f"trial-value source {gain.source!r} required by gain {gain.name!r} is missing"
            )
        values = data.trial_values[gain.source]
        if values.size < data.n_trials:
            raise ValueError(
                f"gain source {gain.source!r} has {values.size} values for "
                f"{data.n_trials} indexed trials"
            )
        gain_by_time[gain.name] = values[data.trial_index]

    return PreparedDesign(
        spec=spec,
        data=data,
        base_blocks=blocks,
        gain_by_time=gain_by_time,
        lags=lags,
        bases=bases,
        layout=_parameter_layout(spec, bases),
        fit_mask=mask,
    )

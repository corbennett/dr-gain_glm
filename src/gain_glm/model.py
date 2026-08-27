"""Public model specifications and fitted-model results.

The objects in this module describe *what* to fit.  They deliberately contain
no session arrays: predictors refer to named sources that are resolved when a
model is prepared against :class:`gain_glm.data.ModelData`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dataclass_replace
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from .design import PreparedDesign


def _names(values: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    return tuple(values)


@dataclass(frozen=True)
class Event:
    """An event-time input convolved with a temporal kernel."""

    name: str
    window: tuple[float, float]
    n_basis: int = 8
    basis: str = "cosine"
    source: str | None = None
    gains: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", self.name if self.source is None else self.source)
        object.__setattr__(self, "gains", _names(self.gains))
        object.__setattr__(self, "groups", _names(self.groups))
        _validate_predictor(self)


@dataclass(frozen=True)
class Signal:
    """A continuous input convolved with a temporal kernel."""

    name: str
    window: tuple[float, float]
    n_basis: int = 8
    basis: str = "cosine"
    source: str | None = None
    gains: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    align: Literal["interp", "bin"] = "interp"
    normalize: Literal["none", "center", "zscore"] = "none"
    outlier_zscore: float | None = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", self.name if self.source is None else self.source)
        object.__setattr__(self, "gains", _names(self.gains))
        object.__setattr__(self, "groups", _names(self.groups))
        _validate_predictor(self)
        if self.align not in {"interp", "bin"}:
            raise ValueError("align must be 'interp' or 'bin'")
        if self.normalize not in {"none", "center", "zscore"}:
            raise ValueError("normalize must be 'none', 'center', or 'zscore'")


@dataclass(frozen=True)
class History:
    """A causal target-history term built separately for each fitted target."""

    name: str = "history"
    window: tuple[float, float] = (0.01, 1.0)
    n_basis: int = 10
    basis: str = "log_cosine"
    gains: tuple[str, ...] = ()
    groups: tuple[str, ...] = ("history",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "gains", _names(self.gains))
        object.__setattr__(self, "groups", _names(self.groups))
        _validate_predictor(self)


Predictor = Event | Signal | History


def _validate_predictor(predictor: Predictor) -> None:
    if not predictor.name:
        raise ValueError("predictor names cannot be empty")
    if len(predictor.window) != 2 or predictor.window[1] < predictor.window[0]:
        raise ValueError(
            f"invalid window for {predictor.name!r}: {predictor.window!r}"
        )
    if predictor.n_basis < 1:
        raise ValueError(f"n_basis must be positive for {predictor.name!r}")
    if len(set(predictor.gains)) != len(predictor.gains):
        raise ValueError(f"duplicate gains on predictor {predictor.name!r}")


@dataclass(frozen=True)
class Gain:
    """A named per-trial value that may modulate one or more predictors."""

    name: str
    source: str | None = None
    groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("gain names cannot be empty")
        object.__setattr__(self, "source", self.name if self.source is None else self.source)
        object.__setattr__(self, "groups", _names(self.groups))


@dataclass(frozen=True)
class ModelSpec:
    """A reusable, data-independent bilinear model declaration."""

    predictors: tuple[Predictor, ...]
    gains: tuple[Gain, ...] = ()
    name: str = "model"

    def __post_init__(self) -> None:
        object.__setattr__(self, "predictors", tuple(self.predictors))
        object.__setattr__(self, "gains", tuple(self.gains))
        if not self.predictors:
            raise ValueError("a model must contain at least one predictor")

        predictor_names = [p.name for p in self.predictors]
        gain_names = [g.name for g in self.gains]
        if len(set(predictor_names)) != len(predictor_names):
            raise ValueError("predictor names must be unique")
        if len(set(gain_names)) != len(gain_names):
            raise ValueError("gain names must be unique")

        known_gains = set(gain_names)
        for predictor in self.predictors:
            unknown = set(predictor.gains) - known_gains
            if unknown:
                raise ValueError(
                    f"predictor {predictor.name!r} references unknown gains: "
                    f"{sorted(unknown)}"
                )

    @property
    def predictor_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.predictors)

    @property
    def gain_names(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.gains)

    def predictor(self, name: str) -> Predictor:
        for predictor in self.predictors:
            if predictor.name == name:
                return predictor
        raise KeyError(f"unknown predictor {name!r}")

    def without(self, *names: str, name: str | None = None) -> ModelSpec:
        """Return a model without the named predictors."""
        requested = set(names)
        unknown = requested - set(self.predictor_names)
        if unknown:
            raise KeyError(f"unknown predictors: {sorted(unknown)}")
        remaining = tuple(p for p in self.predictors if p.name not in requested)
        return ModelSpec(remaining, self.gains, name=name or self.name)

    def without_group(self, group: str, *, name: str | None = None) -> ModelSpec:
        """Return a model without predictors tagged with ``group``."""
        matched = tuple(p.name for p in self.predictors if group in p.groups)
        if not matched:
            raise KeyError(f"no predictors belong to group {group!r}")
        return self.without(*matched, name=name)

    def replace(self, predictor: str, **changes: object) -> ModelSpec:
        """Return a model with one predictor replaced using dataclass fields."""
        current = self.predictor(predictor)
        replacement = dataclass_replace(current, **changes)
        terms = tuple(replacement if p.name == predictor else p for p in self.predictors)
        return ModelSpec(terms, self.gains, name=self.name)

    def add(self, *predictors: Predictor, name: str | None = None) -> ModelSpec:
        return ModelSpec(self.predictors + tuple(predictors), self.gains, name or self.name)


@dataclass(frozen=True)
class FitConfig:
    """Numerical options for one full or reduced fit."""

    regularizer: Literal["ridge", "lasso"] = "ridge"
    kernel_alpha: float | None = None
    gain_alpha: float | None = None
    alphas: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.logspace(-3, 3, 25))
    )
    inner_cv_folds: int | None = None
    max_iter: int = 100
    tol: float = 1e-3
    patience: int = 3
    verbose: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "alphas", tuple(float(a) for a in self.alphas))
        if self.regularizer not in {"ridge", "lasso"}:
            raise ValueError("regularizer must be 'ridge' or 'lasso'")
        if not self.alphas or any(a <= 0 for a in self.alphas):
            raise ValueError("alphas must contain positive values")
        if self.max_iter < 1 or self.patience < 1 or self.tol < 0:
            raise ValueError("invalid ALS convergence settings")


@dataclass(frozen=True)
class Iteration:
    number: int
    mse: float
    kernel_alpha: float
    gain_alpha: float | None


@dataclass(frozen=True)
class FitState:
    """Numerical parameters returned by the pure ALS solver."""

    beta: np.ndarray
    gain: np.ndarray
    intercept: float
    iterations: tuple[Iteration, ...]


@dataclass(frozen=True)
class FittedModel:
    """A prepared design paired with fitted, immutable parameters."""

    prepared: PreparedDesign
    state: FitState
    config: FitConfig
    fit_mask: np.ndarray

    @property
    def spec(self) -> ModelSpec:
        return self.prepared.spec

    @property
    def n_iter(self) -> int:
        return len(self.state.iterations)

    @property
    def converged(self) -> bool:
        return self.n_iter < self.config.max_iter

    @property
    def intercept(self) -> float:
        return float(self.state.intercept)

    def predict(self, *, history: np.ndarray | None = None) -> np.ndarray:
        from ._solver import predict_state

        blocks = self.prepared.blocks_for_target(history)
        return predict_state(self.prepared, blocks, self.state)

    def score(self, y: np.ndarray, mask: np.ndarray | None = None) -> float:
        from ._solver import r2_score

        values = np.asarray(y, dtype=float).ravel()
        used = self.fit_mask if mask is None else np.asarray(mask, dtype=bool).ravel()
        prediction = self.predict(history=values if self.prepared.has_history else None)
        return r2_score(values[used], prediction[used])

    def lags(self, predictor: str) -> np.ndarray:
        return self.prepared.lags[predictor].copy()

    def lags_seconds(self, predictor: str) -> np.ndarray:
        return self.lags(predictor) * self.prepared.data.dt

    def kernel(self, predictor: str) -> np.ndarray:
        sl = self.prepared.layout.beta_slices[predictor]
        return self.prepared.bases[predictor] @ self.state.beta[sl]

    def kernels(self) -> dict[str, np.ndarray]:
        return {p.name: self.kernel(p.name) for p in self.spec.predictors}

    def gain_offset(self, predictor: str) -> float:
        index = self.prepared.layout.gain_offsets[predictor]
        return float(self.state.gain[index])

    def gain_coefficient(self, gain: str, predictor: str) -> float:
        index = self.prepared.layout.gain_coefficients[(gain, predictor)]
        return float(self.state.gain[index])

    def gain_table(self) -> Mapping[str, Mapping[str, float]]:
        table: dict[str, dict[str, float]] = {}
        for predictor in self.spec.predictors:
            if not predictor.gains:
                continue
            row = {"offset": self.gain_offset(predictor.name)}
            row.update(
                {
                    gain: self.gain_coefficient(gain, predictor.name)
                    for gain in predictor.gains
                }
            )
            table[predictor.name] = row
        return table

"""Public model specifications and fitted-model results.

The objects in this module describe *what* to fit.  They deliberately contain
no session arrays: predictors refer to named sources that are resolved when a
model is prepared against :class:`gain_glm.data.ModelData`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import TYPE_CHECKING, Literal

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
        object.__setattr__(
            self, "source", self.name if self.source is None else self.source
        )
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
    orthogonalize_against: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source", self.name if self.source is None else self.source
        )
        object.__setattr__(self, "gains", _names(self.gains))
        object.__setattr__(self, "groups", _names(self.groups))
        _validate_predictor(self)
        if self.align not in {"interp", "bin"}:
            raise ValueError("align must be 'interp' or 'bin'")
        if self.normalize not in {"none", "center", "zscore"}:
            raise ValueError("normalize must be 'none', 'center', or 'zscore'")
        if self.outlier_zscore is not None and self.outlier_zscore <= 0:
            raise ValueError("outlier_zscore must be positive or None")
        if self.orthogonalize_against is not None and (
            not isinstance(self.orthogonalize_against, str)
            or not self.orthogonalize_against
        ):
            raise ValueError("orthogonalize_against must be a predictor name or None")


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
        raise ValueError(f"invalid window for {predictor.name!r}: {predictor.window!r}")
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
        object.__setattr__(
            self, "source", self.name if self.source is None else self.source
        )
        object.__setattr__(self, "groups", _names(self.groups))


@dataclass(frozen=True)
class ResolvedDropout:
    name: str
    predictors: tuple[str, ...]
    gains: tuple[str, ...]
    refit: Literal["gains", "full"]
    gain_terms: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Dropout:
    """A named reduced-model comparison attached to a model specification."""

    name: str
    remove_predictors: tuple[str, ...] = ()
    remove_gains: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    refit: Literal["auto", "gains", "full"] = "auto"
    remove_gain_terms: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "remove_predictors", tuple(self.remove_predictors))
        object.__setattr__(self, "remove_gains", tuple(self.remove_gains))
        gain_terms: list[tuple[str, str]] = []
        for term in self.remove_gain_terms:
            if isinstance(term, str):
                raise TypeError("gain terms must be (gain, predictor) pairs")
            pair = tuple(term)
            if not all(isinstance(name, str) for name in pair):
                raise TypeError("gain and predictor names must be strings")
            if len(pair) != 2:
                raise ValueError("gain terms must be (gain, predictor) pairs")
            gain_terms.append(pair)
        object.__setattr__(self, "remove_gain_terms", tuple(gain_terms))
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
    def gain_terms(
        cls,
        gain: str,
        *predictors: str,
        name: str | None = None,
        refit: Literal["auto", "gains", "full"] = "auto",
    ) -> Dropout:
        """Remove one gain's coefficient from selected predictors only."""
        if not predictors:
            raise ValueError("gain_terms requires at least one predictor")
        label = name or f"{gain}:" + "+".join(predictors)
        return cls(
            label,
            remove_gain_terms=tuple((gain, predictor) for predictor in predictors),
            refit=refit,
        )

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
        gain_terms: Sequence[tuple[str, str]] = (),
        groups: Sequence[str] = (),
        refit: Literal["auto", "gains", "full"] = "auto",
    ) -> Dropout:
        """Declare a mixed reduction using any combination of term selectors."""
        return cls(
            name,
            remove_predictors=tuple(predictors),
            remove_gains=tuple(gains),
            remove_gain_terms=tuple(gain_terms),
            groups=tuple(groups),
            refit=refit,
        )

    def resolve_for(self, spec: ModelSpec) -> ResolvedDropout:
        """Validate and resolve this comparison against a model declaration."""
        known_predictors = set(spec.predictor_names)
        known_gains = set(spec.gain_names)
        ordered_gain_terms = tuple(
            (gain, predictor.name)
            for predictor in spec.predictors
            for gain in predictor.gains
        )
        known_gain_terms = set(ordered_gain_terms)
        unknown_predictors = set(self.remove_predictors) - known_predictors
        unknown_gains = set(self.remove_gains) - known_gains
        unknown_gain_terms = set(self.remove_gain_terms) - known_gain_terms
        if unknown_predictors:
            raise ValueError(
                f"dropout {self.name!r} has unknown predictors: "
                f"{sorted(unknown_predictors)}"
            )
        if unknown_gains:
            raise ValueError(
                f"dropout {self.name!r} has unknown gains: {sorted(unknown_gains)}"
            )
        if unknown_gain_terms:
            raise ValueError(
                f"dropout {self.name!r} has unknown gain terms: "
                f"{sorted(unknown_gain_terms)}"
            )

        predictors = set(self.remove_predictors)
        for group in self.groups:
            matches = set(spec.group_members(group))
            if not matches:
                raise ValueError(
                    f"dropout {self.name!r}: no predictors belong to group {group!r}"
                )
            predictors.update(matches)
        if not predictors and not self.remove_gains and not self.remove_gain_terms:
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
            predictors=tuple(p for p in spec.predictor_names if p in predictors),
            gains=tuple(g for g in spec.gain_names if g in self.remove_gains),
            gain_terms=tuple(
                term for term in ordered_gain_terms if term in self.remove_gain_terms
            ),
            refit=strategy,
        )

    def resolve(self, prepared: PreparedDesign) -> ResolvedDropout:
        """Resolve this comparison against a prepared model design."""
        return self.resolve_for(prepared.spec)


@dataclass(frozen=True)
class ModelSpec:
    """A reusable, data-independent bilinear model declaration."""

    predictors: tuple[Predictor, ...]
    gains: tuple[Gain, ...] = ()
    name: str = "model"
    dt: float = field(kw_only=True)
    fit_window: tuple[float, float] | None = field(default=None, kw_only=True)
    fit_events: tuple[str, ...] = field(default=(), kw_only=True)
    dropouts: tuple[Dropout, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "predictors", tuple(self.predictors))
        object.__setattr__(self, "gains", tuple(self.gains))
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "fit_events", _names(self.fit_events))
        object.__setattr__(self, "dropouts", tuple(self.dropouts))
        if not self.predictors:
            raise ValueError("a model must contain at least one predictor")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.fit_window is None:
            if self.fit_events:
                raise ValueError("fit_events requires fit_window")
        else:
            if len(self.fit_window) != 2:
                raise ValueError(f"invalid fit_window: {self.fit_window!r}")
            fit_window = tuple(float(value) for value in self.fit_window)
            if not all(np.isfinite(fit_window)) or fit_window[1] < fit_window[0]:
                raise ValueError(f"invalid fit_window: {self.fit_window!r}")
            if not self.fit_events:
                raise ValueError("fit_window requires at least one fit event")
            object.__setattr__(self, "fit_window", fit_window)
        if any(not event for event in self.fit_events):
            raise ValueError("fit event names cannot be empty")
        if len(set(self.fit_events)) != len(self.fit_events):
            raise ValueError("fit event names must be unique")

        predictor_names = [p.name for p in self.predictors]
        gain_names = [g.name for g in self.gains]
        if len(set(predictor_names)) != len(predictor_names):
            raise ValueError("predictor names must be unique")
        if len(set(gain_names)) != len(gain_names):
            raise ValueError("gain names must be unique")

        predictors_by_name = {
            predictor.name: predictor for predictor in self.predictors
        }
        orthogonalization_dependencies: dict[str, str] = {}
        for predictor in self.predictors:
            if not isinstance(predictor, Signal):
                continue
            reference_name = predictor.orthogonalize_against
            if reference_name is None:
                continue
            if reference_name not in predictors_by_name:
                raise ValueError(
                    f"signal {predictor.name!r} is orthogonalized against unknown "
                    f"predictor {reference_name!r}"
                )
            if reference_name == predictor.name:
                raise ValueError(
                    f"signal {predictor.name!r} cannot be orthogonalized against itself"
                )
            if not isinstance(predictors_by_name[reference_name], Signal):
                raise TypeError(
                    f"signal {predictor.name!r} cannot be orthogonalized against "
                    f"non-signal predictor {reference_name!r}"
                )
            orthogonalization_dependencies[predictor.name] = reference_name

        for name in orthogonalization_dependencies:
            path: set[str] = set()
            current = name
            while current in orthogonalization_dependencies:
                if current in path:
                    raise ValueError(
                        "signal orthogonalization dependencies contain a cycle"
                    )
                path.add(current)
                current = orthogonalization_dependencies[current]

        known_gains = set(gain_names)
        used_gains: set[str] = set()
        for predictor in self.predictors:
            unknown = set(predictor.gains) - known_gains
            if unknown:
                raise ValueError(
                    f"predictor {predictor.name!r} references unknown gains: "
                    f"{sorted(unknown)}"
                )
            used_gains.update(predictor.gains)
        unused = known_gains - used_gains
        if unused:
            raise ValueError(f"gains are not used by any predictor: {sorted(unused)}")

        if not all(isinstance(dropout, Dropout) for dropout in self.dropouts):
            raise TypeError("dropouts must contain Dropout instances")
        dropout_names = [dropout.name for dropout in self.dropouts]
        if len(set(dropout_names)) != len(dropout_names):
            raise ValueError("dropout names must be unique")
        for dropout in self.dropouts:
            dropout.resolve_for(self)

    @property
    def predictor_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.predictors)

    @property
    def gain_names(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.gains)

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(sorted({group for p in self.predictors for group in p.groups}))

    def group_members(self, group: str) -> tuple[str, ...]:
        """Predictor names carrying ``group``, in model declaration order."""
        return tuple(p.name for p in self.predictors if group in p.groups)

    def predictor(self, name: str) -> Predictor:
        for predictor in self.predictors:
            if predictor.name == name:
                return predictor
        raise KeyError(f"unknown predictor {name!r}")

    def without(
        self,
        *names: str,
        name: str | None = None,
        dropouts: Sequence[Dropout] | None = None,
    ) -> ModelSpec:
        """Return a model without the named predictors."""
        requested = set(names)
        unknown = requested - set(self.predictor_names)
        if unknown:
            raise KeyError(f"unknown predictors: {sorted(unknown)}")
        remaining = tuple(p for p in self.predictors if p.name not in requested)
        return ModelSpec(
            remaining,
            self.gains,
            name=name or self.name,
            dt=self.dt,
            fit_window=self.fit_window,
            fit_events=self.fit_events,
            dropouts=self.dropouts if dropouts is None else tuple(dropouts),
        )

    def without_group(
        self,
        group: str,
        *,
        name: str | None = None,
        dropouts: Sequence[Dropout] | None = None,
    ) -> ModelSpec:
        """Return a model without predictors tagged with ``group``."""
        matched = self.group_members(group)
        if not matched:
            raise KeyError(f"no predictors belong to group {group!r}")
        return self.without(*matched, name=name, dropouts=dropouts)

    def replace(
        self,
        predictor: str,
        *,
        dropouts: Sequence[Dropout] | None = None,
        **changes: object,
    ) -> ModelSpec:
        """Return a model with one predictor replaced using dataclass fields."""
        current = self.predictor(predictor)
        replacement = dataclass_replace(current, **changes)
        terms = tuple(
            replacement if p.name == predictor else p for p in self.predictors
        )
        return ModelSpec(
            terms,
            self.gains,
            name=self.name,
            dt=self.dt,
            fit_window=self.fit_window,
            fit_events=self.fit_events,
            dropouts=self.dropouts if dropouts is None else tuple(dropouts),
        )

    def add(
        self,
        *predictors: Predictor,
        name: str | None = None,
        dropouts: Sequence[Dropout] | None = None,
    ) -> ModelSpec:
        return ModelSpec(
            self.predictors + tuple(predictors),
            self.gains,
            name or self.name,
            dt=self.dt,
            fit_window=self.fit_window,
            fit_events=self.fit_events,
            dropouts=self.dropouts if dropouts is None else tuple(dropouts),
        )


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
    relative_mse_change: float
    relative_prediction_change: float
    relative_gain_change: float
    max_abs_gain_change: float


@dataclass(frozen=True)
class ConvergenceDiagnostics:
    """Final numerical diagnostics from one alternating fit."""

    converged: bool
    n_iter: int
    relative_mse_change: float
    relative_prediction_change: float
    relative_gain_change: float
    max_abs_gain_change: float

    def to_dict(self) -> dict[str, bool | int | float]:
        return {
            "converged": self.converged,
            "n_iter": self.n_iter,
            "relative_mse_change": self.relative_mse_change,
            "relative_prediction_change": self.relative_prediction_change,
            "relative_gain_change": self.relative_gain_change,
            "max_abs_gain_change": self.max_abs_gain_change,
        }


@dataclass(frozen=True)
class FitState:
    """Numerical parameters returned by the pure ALS solver."""

    beta: np.ndarray
    gain: np.ndarray
    intercept: float
    iterations: tuple[Iteration, ...]
    converged: bool

    def __post_init__(self) -> None:
        beta = np.array(self.beta, dtype=float, copy=True).ravel()
        gain = np.array(self.gain, dtype=float, copy=True).ravel()
        beta.setflags(write=False)
        gain.setflags(write=False)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "iterations", tuple(self.iterations))

    @property
    def diagnostics(self) -> ConvergenceDiagnostics:
        last = self.iterations[-1]
        return ConvergenceDiagnostics(
            converged=self.converged,
            n_iter=len(self.iterations),
            relative_mse_change=last.relative_mse_change,
            relative_prediction_change=last.relative_prediction_change,
            relative_gain_change=last.relative_gain_change,
            max_abs_gain_change=last.max_abs_gain_change,
        )


@dataclass(frozen=True)
class FittedModel:
    """A prepared design paired with fitted, immutable parameters."""

    prepared: PreparedDesign
    state: FitState
    config: FitConfig
    fit_mask: np.ndarray

    def __post_init__(self) -> None:
        mask = np.array(self.fit_mask, dtype=bool, copy=True).ravel()
        if mask.size != self.prepared.data.n_time:
            raise ValueError("fit mask has the wrong length")
        mask.setflags(write=False)
        object.__setattr__(self, "fit_mask", mask)

    @property
    def spec(self) -> ModelSpec:
        return self.prepared.spec

    @property
    def n_iter(self) -> int:
        return len(self.state.iterations)

    @property
    def converged(self) -> bool:
        return self.state.converged

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
        if values.size != self.prepared.data.n_time:
            raise ValueError(
                f"target has length {values.size}, expected {self.prepared.data.n_time}"
            )
        if used.size != values.size:
            raise ValueError(f"mask has length {used.size}, expected {values.size}")
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

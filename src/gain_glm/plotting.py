"""Small plotting helpers kept outside the fitting API."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .evaluation import EvaluationResult
from .model import FittedModel


def plot_kernels(fitted: FittedModel, predictors: Sequence[str] | None = None):
    """Plot time-domain kernels and return ``(figure, axes)``."""
    import matplotlib.pyplot as plt

    names = list(predictors or fitted.spec.predictor_names)
    figure, axes = plt.subplots(
        len(names), 1, squeeze=False, figsize=(7, 2.2 * len(names))
    )
    for axis, name in zip(axes[:, 0], names):
        axis.plot(fitted.lags_seconds(name), fitted.kernel(name))
        axis.axvline(0, color="0.7", linewidth=1)
        axis.set(title=name, xlabel="lag (s)", ylabel="kernel")
    figure.tight_layout()
    return figure, axes[:, 0]


def plot_prediction(
    result: EvaluationResult,
    y: np.ndarray,
    *,
    time: np.ndarray | None = None,
    mask: np.ndarray | None = None,
):
    """Plot the full-data target and fitted prediction."""
    import matplotlib.pyplot as plt

    values = np.asarray(y, dtype=float).ravel()
    x = np.arange(values.size) * result.fit.prepared.data.dt if time is None else time
    used = (
        np.ones(values.size, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    prediction = result.fit.predict(
        history=values if result.fit.prepared.has_history else None
    )
    figure, axis = plt.subplots(figsize=(11, 3))
    axis.plot(x[used], values[used], label="observed", linewidth=1)
    axis.plot(x[used], prediction[used], label="fitted", linewidth=1)
    axis.set(xlabel="time (s)", ylabel="target")
    axis.legend()
    figure.tight_layout()
    return figure, axis

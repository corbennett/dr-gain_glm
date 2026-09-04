"""Data containers and time-grid helpers for gain GLMs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


def _readonly(values: np.ndarray, *, dtype=float) -> np.ndarray:
    output = np.array(values, dtype=dtype, copy=True).ravel()
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class TimedSignal:
    """Continuous values, optionally paired with sample times in seconds."""

    values: np.ndarray
    times: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = _readonly(self.values)
        times = None if self.times is None else _readonly(self.times)
        if times is not None and times.size != values.size:
            raise ValueError(
                f"signal values and times must have equal length, got "
                f"{values.size} and {times.size}"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "times", times)


@dataclass(frozen=True)
class ModelData:
    """Named model inputs sampled on one shared target time grid.

    Event and signal times are relative to the beginning of this grid. Trial
    indices must be non-negative and index every per-trial value used by the
    model.
    """

    dt: float
    trial_index: np.ndarray
    events: Mapping[str, np.ndarray] = field(default_factory=dict)
    signals: Mapping[str, TimedSignal | np.ndarray] = field(default_factory=dict)
    trial_values: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        trial_index = _readonly(self.trial_index, dtype=int)
        if trial_index.size == 0:
            raise ValueError("trial_index cannot be empty")
        if np.any(trial_index < 0):
            raise ValueError("trial_index must contain only non-negative indices")

        events = {name: _readonly(times) for name, times in self.events.items()}
        signals = {
            name: value if isinstance(value, TimedSignal) else TimedSignal(value)
            for name, value in self.signals.items()
        }
        trial_values = {
            name: _readonly(values) for name, values in self.trial_values.items()
        }
        object.__setattr__(self, "trial_index", trial_index)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "trial_values", trial_values)

    @property
    def n_time(self) -> int:
        return self.trial_index.size

    @property
    def n_trials(self) -> int:
        return int(self.trial_index.max()) + 1


class TimeBins(NamedTuple):
    edges: np.ndarray
    centers: np.ndarray
    n_time: int
    dt: float
    start: float
    end: float


def make_time_bins(start: float, end: float, dt: float) -> TimeBins:
    """Return a fixed-width grid covering complete bins in ``[start, end)``."""
    if end <= start:
        raise ValueError("end must be greater than start")
    if dt <= 0:
        raise ValueError("dt must be positive")
    n_time = int(np.floor((end - start) / dt))
    edges = start + np.arange(n_time + 1) * dt
    return TimeBins(
        edges=edges,
        centers=edges[:-1] + dt / 2,
        n_time=n_time,
        dt=float(dt),
        start=float(start),
        end=float(end),
    )


def bin_spike_times(
    spike_times: np.ndarray,
    start: float,
    end: float,
    dt: float,
    *,
    smooth_sigma: float | None = None,
    output: str = "counts",
) -> np.ndarray:
    """Bin spike times onto a model grid, optionally smoothing or returning Hz."""
    if output not in {"counts", "rate"}:
        raise ValueError("output must be 'counts' or 'rate'")
    bins = make_time_bins(start, end, dt)
    y = np.zeros(bins.n_time)
    if spike_times is not None:
        relative = np.asarray(spike_times, dtype=float).ravel() - start
        indices = np.floor(relative / dt).astype(int)
        indices = indices[(indices >= 0) & (indices < bins.n_time)]
        np.add.at(y, indices, 1.0)
    if smooth_sigma is not None and smooth_sigma > 0:
        sigma_bins = smooth_sigma / dt
        radius = max(1, int(np.ceil(4 * sigma_bins)))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        y = np.convolve(y, kernel / kernel.sum(), mode="same")
    return y / dt if output == "rate" else y


def make_trial_index(
    starts: np.ndarray,
    ends: np.ndarray,
    dt: float,
    *,
    n_time: int | None = None,
) -> np.ndarray:
    """Map each time bin to its containing trial.

    Uncovered or overlapping bins are rejected. This keeps gain broadcasting
    and trial-held-out cross-validation unambiguous.
    """
    starts = np.asarray(starts, dtype=float).ravel()
    ends = np.asarray(ends, dtype=float).ravel()
    if starts.size != ends.size:
        raise ValueError("starts and ends must have equal length")
    if starts.size == 0 or np.any(ends <= starts):
        raise ValueError("trials must be non-empty and each end must exceed its start")
    if n_time is None:
        n_time = int(np.floor(ends.max() / dt))

    trial_index = np.full(n_time, -1, dtype=int)
    for trial, (start, end) in enumerate(zip(starts, ends)):
        first = max(0, int(np.floor(start / dt)))
        last = min(n_time, int(np.floor(end / dt)))
        if np.any(trial_index[first:last] >= 0):
            raise ValueError(f"trial {trial} overlaps an earlier trial")
        trial_index[first:last] = trial
    if np.any(trial_index < 0):
        first = int(np.flatnonzero(trial_index < 0)[0])
        raise ValueError(f"time bin {first} is not covered by a trial")
    return trial_index


def _window_bin_bounds(
    window: tuple[float, float], dt: float
) -> tuple[int, int]:
    """Return the first bin and exclusive end edge for a time window.

    The lower bound is the beginning of the first selected bin. The upper
    bound is the end of the final selected bin, so it becomes an exclusive
    bin edge after rounding to the model grid. A zero-width window retains
    the package's existing one-bin shorthand.
    """
    if window[1] < window[0]:
        raise ValueError(f"invalid window: {window!r}")
    low = int(np.floor(window[0] / dt))
    high = int(np.ceil(window[1] / dt))
    if high == low:
        high = low + 1
    return low, high


def windows_mask(
    event_times: np.ndarray,
    n_time: int,
    dt: float,
    window: tuple[float, float] = (-1.0, 2.0),
) -> np.ndarray:
    """Mark bins whose window runs from its lower bound to upper edge."""
    mask = np.zeros(n_time, dtype=bool)
    if event_times is None:
        return mask
    low, high = _window_bin_bounds(window, dt)
    for event_bin in np.floor(np.asarray(event_times) / dt).astype(int):
        first = max(0, event_bin + low)
        last = min(n_time, event_bin + high)
        mask[first:last] = True
    return mask

"""A repository of declarative session designs for the bilinear gain GLM.

A *design* is just a list of kernel specs plus a gain spec. Each kernel spec
says (a) where its data comes from — a small `source` function over the loaded
`SessionData` — and (b) its kernel parameters (window, n_basis, ...). A generic
builder turns a spec list into predictors and hands them to
`run_fit.finalize_design`, so adding a new event OR continuous kernel is a
single line: append a spec. No copying of a builder function.

    EventKernel(name, source, window, n_basis, gain_modulated)
        source(data) -> absolute event-time array (made task-relative for you)
    ContinuousKernel(name, source, window, n_basis, normalize, gain_modulated)
        source(data) -> (times, values); times=None means one value per bin

`source` helpers (trials_col, field, running, pupil, lp_feature, ...) cover the
common streams; write your own `lambda data: (...)` for anything else. The
context gain automatically modulates exactly the kernels flagged
`gain_modulated=True`, so it tracks the kernel list.

Designs are registered in DESIGNS (name -> design_fn). A design_fn has the
signature compare_models expects: `design_fn(SessionData) -> design dict`.

    from designs import DESIGNS
    DESIGNS["default"](data)            # reproduces run_fit.assemble_design
    DESIGNS["all_response"](data)       # an alternative
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

from bilinear_glm import ContinuousPredictor, EventPredictor, GainModulator
from run_fit import STIM_COLUMNS, SessionData, finalize_design

# ---------------------------------------------------------------------------
# source helpers: SessionData -> raw inputs for a kernel
# ---------------------------------------------------------------------------
def trials_col(col: str) -> Callable[[SessionData], np.ndarray]:
    """Event times = stim_start_time of trials where `col` is True."""
    def source(data: SessionData) -> np.ndarray:
        return (data.trials_df.filter(pl.col(col))
                .select("stim_start_time")["stim_start_time"]
                .drop_nulls().to_numpy())
    return source


def field(name: str) -> Callable[[SessionData], np.ndarray]:
    """Event times from a SessionData field (e.g. 'lick_times', 'reward_times')."""
    return lambda data: getattr(data, name)


def running(data: SessionData):
    return data.running_speed[:, 0], data.running_speed[:, 1]


def pupil(data: SessionData):
    return data.pupil[:, 0], data.pupil[:, 1]


def lp_feature(feature: str, *, likelihood_min: float = 0.98,
               jitter_sd: float = 3.0) -> Callable[[SessionData], tuple]:
    """Lightning-pose feature: euclidean position, likelihood/jitter-filtered."""
    def source(data: SessionData):
        x = data.lp.select(pl.col(feature + "_x")).to_numpy().flatten()
        y = data.lp.select(pl.col(feature + "_y")).to_numpy().flatten()
        like = data.lp.select(pl.col(feature + "_likelihood")).to_numpy().flatten()
        tnorm = data.lp.select(pl.col(feature + "_temporal_norm")).to_numpy().flatten()
        mask = (like > likelihood_min) & (
            tnorm < np.nanmean(tnorm) + jitter_sd * np.nanstd(tnorm))
        return data.side_frame_times[mask], np.sqrt(x ** 2 + y ** 2)[mask]
    return source


def context_baseline(data: SessionData):
    """One value per trial (+/-1 context label), placed at the trial start."""
    return data.trial_start_times, data.trial_context_labels


def session_time(data: SessionData):
    """A linear 0->1 ramp over the session, one value per bin (times=None)."""
    return None, np.arange(data.T) / data.T


def context_values(data: SessionData) -> np.ndarray:
    """Per-trial gain values for the context modulator (+1 vis, -1 aud)."""
    return data.trial_context_labels


# ---------------------------------------------------------------------------
# kernel specs
# ---------------------------------------------------------------------------
@dataclass
class EventKernel:
    name: str
    source: Callable[[SessionData], np.ndarray]
    window: tuple = (0.0, 1.0)
    n_basis: int = 10
    basis: str = "cosine"
    gain_modulated: bool = False

    def build(self, data: SessionData) -> EventPredictor:
        times = np.asarray(self.source(data), dtype=float) - data.task_start_time
        return EventPredictor(self.name, times, window=self.window,
                              n_basis=self.n_basis, basis=self.basis,
                              gain_modulated=self.gain_modulated)


@dataclass
class ContinuousKernel:
    name: str
    source: Callable[[SessionData], tuple]  # -> (times|None, values)
    window: tuple = (-1.0, 1.0)
    n_basis: int = 10
    basis: str = "cosine"
    normalize: str = "zscore"
    gain_modulated: bool = False

    def build(self, data: SessionData) -> ContinuousPredictor:
        times, values = self.source(data)
        if times is not None:
            times = np.asarray(times, dtype=float) - data.task_start_time
        return ContinuousPredictor(self.name, values=np.asarray(values, dtype=float),
                                   window=self.window, n_basis=self.n_basis,
                                   basis=self.basis, gain_modulated=self.gain_modulated,
                                   times=times, normalize=self.normalize)


def make_design_fn(kernels, *, gain_name: str = "context",
                   gain_source: Callable = context_values,
                   **finalize_kwargs):
    """Turn a kernel-spec list into a design_fn(SessionData) -> design dict.

    The gain `gain_name` modulates exactly the kernels with gain_modulated=True.
    `finalize_kwargs` pass through to run_fit.finalize_design (kernel_regularizer,
    alphas, stim_fit_window, cv_folds, ...).
    """
    def design_fn(data: SessionData) -> dict:
        predictors = [k.build(data) for k in kernels]
        modulated = [k.name for k in kernels if k.gain_modulated]
        gains = ([GainModulator(gain_name, values=gain_source(data), modulates=modulated)]
                 if modulated else [])
        return finalize_design(data, predictors, gains, **finalize_kwargs)
    return design_fn


# ---------------------------------------------------------------------------
# Reusable kernel groups — compose these into designs.
# ---------------------------------------------------------------------------
# Stimulus onsets: short, few bases (the default).
STIM_KERNELS = [
    EventKernel(c, trials_col(c), window=(0, 0.1), n_basis=2, gain_modulated=True)
    for c in STIM_COLUMNS
]

# Behavioral nuisance / continuous predictors shared by most designs.
BEHAVIOR_KERNELS = [
    EventKernel("licks", field("lick_times"), window=(0, 0.2), n_basis=4, gain_modulated=True),
    EventKernel("rewards", field("reward_times"), window=(-0.2, 1), n_basis=12, gain_modulated=True),
    ContinuousKernel("running_speed", running, window=(-1, 1), n_basis=10),
    ContinuousKernel("pupil_area", pupil, window=(-1, 1), n_basis=10),
    ContinuousKernel("ear", lp_feature("ear_base_l"), window=(-1, 1), n_basis=10),
    ContinuousKernel("jaw", lp_feature("jaw"), window=(-1, 1), n_basis=10),
    ContinuousKernel("nose", lp_feature("nose_tip"), window=(-1, 1), n_basis=10),
    ContinuousKernel("whisker_pad", lp_feature("whisker_pad_l_side"), window=(-1, 1), n_basis=10),
    ContinuousKernel("context_baseline", context_baseline, window=(0, 0), n_basis=1, normalize="none"),
    ContinuousKernel("time", session_time, window=(0, 0), n_basis=1, normalize="zscore"),
]

# A single long "hit" response kernel (the current production default).
HIT_KERNEL = [
    EventKernel("is_hit", trials_col("is_hit"), window=(0.1, 1), n_basis=9, gain_modulated=True),
]

# Long stimulus kernels (0-1 s, 10 bases) instead of the short ones.
LONG_STIM_KERNELS = [
    EventKernel(c, trials_col(c), window=(0, 1), n_basis=10, gain_modulated=True)
    for c in STIM_COLUMNS
]

# All four behavioral-outcome response kernels.
RESPONSE_KERNELS = [
    EventKernel(c, trials_col(c), window=(0.1, 1), n_basis=9, gain_modulated=True)
    for c in ("is_hit", "is_miss", "is_correct_reject", "is_false_alarm")
]


# ---------------------------------------------------------------------------
# The registry of named designs. Add entries here.
# ---------------------------------------------------------------------------
DESIGNS: dict[str, Callable[[SessionData], dict]] = {
    # reproduces run_fit.assemble_design (default production design)
    "default": make_design_fn(STIM_KERNELS + HIT_KERNEL + BEHAVIOR_KERNELS),
    # drop the lightning-pose face predictors
    "no_lp": make_design_fn(
        STIM_KERNELS + HIT_KERNEL
        + [k for k in BEHAVIOR_KERNELS if k.name not in {"ear", "jaw", "nose", "whisker_pad"}]),
    # no hit kernel, long stimulus kernels
    "no_hit_long_stim": make_design_fn(LONG_STIM_KERNELS + BEHAVIOR_KERNELS),
    # all four outcome response kernels alongside the short stimulus kernels
    "all_response": make_design_fn(STIM_KERNELS + RESPONSE_KERNELS + BEHAVIOR_KERNELS),
}

# Name each design_fn after its registry key so build logs are informative.
for _name, _fn in DESIGNS.items():
    _fn.__name__ = f"design[{_name}]"

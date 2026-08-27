"""A small, declarative API for bilinear gain GLMs."""

from .data import (
    ModelData,
    TimedSignal,
    TimeBins,
    bin_spike_times,
    make_time_bins,
    make_trial_index,
    windows_mask,
)
from .design import PreparedDesign, compile_design
from .evaluation import (
    CVConfig,
    CVResult,
    Dropout,
    DropoutResult,
    EvaluationResult,
    ResolvedDropout,
    evaluate,
)
from .model import (
    Event,
    FitConfig,
    FittedModel,
    Gain,
    History,
    ModelSpec,
    Signal,
)

__all__ = [
    "CVConfig",
    "CVResult",
    "Dropout",
    "DropoutResult",
    "EvaluationResult",
    "Event",
    "FitConfig",
    "FittedModel",
    "Gain",
    "History",
    "ModelData",
    "ModelSpec",
    "PreparedDesign",
    "ResolvedDropout",
    "Signal",
    "TimeBins",
    "TimedSignal",
    "bin_spike_times",
    "compile_design",
    "evaluate",
    "make_time_bins",
    "make_trial_index",
    "windows_mask",
]

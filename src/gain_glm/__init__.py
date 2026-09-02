"""A small, declarative API for bilinear gain GLMs."""

from .data import (
    ModelData,
    TimeBins,
    TimedSignal,
    bin_spike_times,
    make_time_bins,
    make_trial_index,
    windows_mask,
)
from .design import PreparedDesign, compile_design
from .evaluation import (
    CVConfig,
    CVResult,
    DropoutResult,
    EvaluationResult,
    evaluate,
)
from .model import (
    ConvergenceDiagnostics,
    Dropout,
    Event,
    FitConfig,
    FittedModel,
    Gain,
    History,
    ModelSpec,
    ResolvedDropout,
    Signal,
)

__all__ = [
    "CVConfig",
    "CVResult",
    "ConvergenceDiagnostics",
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

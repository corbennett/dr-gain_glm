"""Dynamic Routing NWB adapter and reusable model declarations.

Everything specific to the experiment's NWB schema lives here. The core GLM
only sees a :class:`ModelData` object containing named arrays.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import lazynwb
import numpy as np
import polars as pl

from .data import (
    ModelData,
    TimedSignal,
    bin_spike_times,
    make_trial_index,
    windows_mask,
)
from .design import PreparedDesign, compile_design
from .model import Dropout, Event, Gain, ModelSpec, Signal

lazynwb.config.anon = True

DEFAULT_DT = 0.025
STIMULUS_FIT_WINDOW = (-0.5, 1.0)
QC_COLUMN = "default_qc"
STIMULUS_EVENTS = (
    "is_aud_target",
    "is_aud_nontarget",
    "is_vis_target",
    "is_vis_nontarget",
)
OUTCOME_EVENTS = (
    "is_hit",
    "is_miss",
    "is_correct_reject",
    "is_false_alarm",
)


def _stimulus_predictors(
    *, window: tuple[float, float] = (0, 0.1), n_basis: int = 4
) -> tuple[Event, ...]:
    return tuple(
        Event(
            name,
            window=window,
            n_basis=n_basis,
            gains=("context",),
            groups=("stimulus", "task"),
        )
        for name in STIMULUS_EVENTS
    )


LATE_STIMULUS_PREDICTOR_NAMES = tuple(f"{source}_late" for source in STIMULUS_EVENTS)


def _late_stimulus_predictors() -> tuple[Event, ...]:
    return tuple(
        Event(
            name,
            source=source,
            window=(0.1, 1),
            n_basis=9,
            gains=("context",),
            groups=("stimulus", "late_stimulus", "task"),
        )
        for name, source in zip(LATE_STIMULUS_PREDICTOR_NAMES, STIMULUS_EVENTS)
    )


BEHAVIOR_PREDICTORS = (
    Event(
        "licks",
        window=(0, 0.2),
        n_basis=4,
        gains=("context",),
        groups=("behavior", "action"),
    ),
    Event(
        "rewards",
        window=(-0.2, 1),
        n_basis=12,
        gains=("context",),
        groups=("behavior", "outcome"),
    ),
    Signal(
        "running_speed",
        window=(-1, 1),
        n_basis=10,
        normalize="zscore",
        groups=("behavior",),
    ),
    Signal(
        "pupil_area",
        window=(-1, 1),
        n_basis=10,
        normalize="zscore",
        groups=("behavior",),
    ),
    Signal(
        "ear",
        window=(-1, 1),
        n_basis=10,
        normalize="zscore",
        groups=("behavior", "face"),
    ),
    Signal(
        "jaw",
        window=(-1, 1),
        n_basis=10,
        normalize="zscore",
        groups=("behavior", "face"),
    ),
    Signal(
        "nose",
        window=(-1, 1),
        n_basis=10,
        normalize="zscore",
        groups=("behavior", "face"),
    ),
    Signal(
        "whisker_pad",
        window=(-1, 1),
        n_basis=10,
        normalize="zscore",
        groups=("behavior", "face"),
    ),
    Signal(
        "context_baseline",
        window=(0, 0),
        n_basis=1,
        groups=("context",),
    ),
    Signal(
        "time",
        window=(0, 0),
        n_basis=1,
        normalize="zscore",
        groups=("nuisance",),
    ),
)

DEFAULT_DROPOUTS = (
    Dropout.gain("context"),
    Dropout.gain_terms(
        "context",
        *STIMULUS_EVENTS,
        name="early_stim_context_gain",
    ),
    Dropout.gain_terms(
        "context",
        *LATE_STIMULUS_PREDICTOR_NAMES,
        name="late_stim_context_gain",
    ),
    Dropout.predictors("context_baseline"),
)

DEFAULT_MODEL = ModelSpec(
    predictors=(
        *_stimulus_predictors(),
        *_late_stimulus_predictors(),
        Event(
            "is_hit",
            window=(0.1, 1),
            n_basis=9,
            gains=("context",),
            groups=("response", "outcome", "task"),
        ),
        *BEHAVIOR_PREDICTORS,
    ),
    gains=(Gain("context", source="trial_context"),),
    name="default",
    dt=DEFAULT_DT,
    fit_window=STIMULUS_FIT_WINDOW,
    fit_events=STIMULUS_EVENTS,
    dropouts=DEFAULT_DROPOUTS,
)

NO_FACE_MODEL = DEFAULT_MODEL.without_group("face", name="no_face")

NO_HIT_LONG_STIM_MODEL = ModelSpec(
    predictors=(*_stimulus_predictors(window=(0, 1), n_basis=10), *BEHAVIOR_PREDICTORS),
    gains=DEFAULT_MODEL.gains,
    name="no_hit_long_stim",
    dt=DEFAULT_DT,
    fit_window=STIMULUS_FIT_WINDOW,
    fit_events=STIMULUS_EVENTS,
    dropouts=(
        Dropout.gain("context"),
        Dropout.predictors("context_baseline"),
    ),
)

ALL_RESPONSE_MODEL = DEFAULT_MODEL.add(
    *(
        Event(
            name,
            window=(0.1, 1),
            n_basis=9,
            gains=("context",),
            groups=("response", "outcome", "task"),
        )
        for name in OUTCOME_EVENTS
        if name != "is_hit"
    ),
    name="all_response",
)

ONLY_BASELINE_MODEL = ModelSpec(
    predictors=(
        Signal(
            "context_baseline",
            window=(0, 0),
            n_basis=1,
            groups=("context",),
        ),
        Signal(
            "ear",
            window=(-0.5, 0.5),
            n_basis=10,
            normalize="zscore",
            groups=("behavior", "face"),
            orthogonalize_against="context_baseline",
        ),
        Signal(
            "jaw",
            window=(-0.5, 0.5),
            n_basis=10,
            normalize="zscore",
            groups=("behavior", "face"),
            orthogonalize_against="context_baseline",
        ),
        Signal(
            "nose",
            window=(-0.5, 0.5),
            n_basis=10,
            normalize="zscore",
            groups=("behavior", "face"),
            orthogonalize_against="context_baseline",
        ),
        Signal(
            "whisker_pad",
            window=(-0.5, 0.5),
            n_basis=10,
            normalize="zscore",
            groups=("behavior", "face"),
            orthogonalize_against="context_baseline",
        ),
        Signal(
            "running_speed",
            window=(-0.5, 0.5),
            n_basis=10,
            normalize="zscore",
            groups=("behavior",),
        ),
        Signal(
            "pupil_area",
            window=(-0.5, 0.5),
            n_basis=10,
            normalize="zscore",
            groups=("behavior",),
        ),
        Signal(
            "time",
            window=(0, 0),
            n_basis=1,
            normalize="zscore",
            groups=("nuisance",),
        ),
    ),
    gains=(),
    name="only_baseline",
    dt=DEFAULT_DT,
    fit_window=(-1.5, 0),
    fit_events=STIMULUS_EVENTS,
    dropouts=(Dropout.predictors("context_baseline"),),
)

MODELS: Mapping[str, ModelSpec] = {
    model.name: model
    for model in (
        DEFAULT_MODEL,
        NO_FACE_MODEL,
        NO_HIT_LONG_STIM_MODEL,
        ALL_RESPONSE_MODEL,
        ONLY_BASELINE_MODEL,
    )
}


@dataclass(frozen=True)
class SessionData:
    """Session bounds and the model-specific inputs loaded from NWB."""

    nwb_path: str
    task_start_time: float
    task_end_time: float
    data: ModelData

    @property
    def dt(self) -> float:
        return self.data.dt

    @property
    def n_time(self) -> int:
        return self.data.n_time


_TRIAL_EVENT_SOURCES = frozenset((*STIMULUS_EVENTS, *OUTCOME_EVENTS))
_FACE_SIGNAL_FEATURES = {
    "ear": "ear_base_l",
    "jaw": "jaw",
    "nose": "nose_tip",
    "whisker_pad": "whisker_pad_l_side",
}
_SUPPORTED_EVENT_SOURCES = _TRIAL_EVENT_SOURCES | {"licks", "rewards"}
_SUPPORTED_SIGNAL_SOURCES = frozenset(
    {"running_speed", "pupil_area", "context_baseline", "time"}
    | _FACE_SIGNAL_FEATURES.keys()
)
_SUPPORTED_TRIAL_VALUE_SOURCES = frozenset({"trial_context"})


def _required_sources(
    models: tuple[ModelSpec, ...],
) -> tuple[set[str], set[str], set[str]]:
    event_sources = {
        predictor.source
        for model in models
        for predictor in model.predictors
        if isinstance(predictor, Event) and predictor.source is not None
    }
    event_sources.update(source for model in models for source in model.fit_events)
    signal_sources = {
        predictor.source
        for model in models
        for predictor in model.predictors
        if isinstance(predictor, Signal) and predictor.source is not None
    }
    trial_value_sources = {
        gain.source
        for model in models
        for gain in model.gains
        if gain.source is not None
    }
    return event_sources, signal_sources, trial_value_sources


def _validate_sources(
    event_sources: set[str],
    signal_sources: set[str],
    trial_value_sources: set[str],
) -> None:
    unknown = {
        "event": event_sources - _SUPPORTED_EVENT_SOURCES,
        "signal": signal_sources - _SUPPORTED_SIGNAL_SOURCES,
        "trial value": trial_value_sources - _SUPPORTED_TRIAL_VALUE_SOURCES,
    }
    details = [f"{kind}s {sorted(names)}" for kind, names in unknown.items() if names]
    if details:
        raise ValueError(
            "unsupported Dynamic Routing model sources: " + "; ".join(details)
        )


def _trial_events(
    trials: pl.DataFrame, task_start_time: float, column: str
) -> np.ndarray:
    if column not in trials.columns:
        return np.zeros(0)
    return (
        trials.filter(pl.col(column).fill_null(False))
        .select("stim_start_time")["stim_start_time"]
        .drop_nulls()
        .to_numpy()
        - task_start_time
    )


def _pose_signal(
    pose: pl.DataFrame,
    side_frame_times: np.ndarray,
    task_start_time: float,
    feature: str,
    *,
    likelihood_min: float = 0.98,
    jitter_sd: float = 3.0,
) -> TimedSignal:
    x = pose.select(feature + "_x").to_numpy().ravel()
    y = pose.select(feature + "_y").to_numpy().ravel()
    likelihood = pose.select(feature + "_likelihood").to_numpy().ravel()
    temporal_norm = pose.select(feature + "_temporal_norm").to_numpy().ravel()
    if side_frame_times.size != x.size:
        raise ValueError(
            f"side camera has {side_frame_times.size} frame times but "
            f"{feature!r} has {x.size} pose samples"
        )
    valid = (likelihood > likelihood_min) & (
        temporal_norm
        <= np.nanmean(temporal_norm) + jitter_sd * np.nanstd(temporal_norm)
    )
    return TimedSignal(
        np.sqrt(x[valid] ** 2 + y[valid] ** 2),
        side_frame_times[valid] - task_start_time,
    )


def load_session(
    nwb_path: str,
    model: ModelSpec,
    *additional_models: ModelSpec,
) -> SessionData:
    """Load the union of session inputs required by the supplied models."""
    models = (model, *additional_models)
    if any(not np.isclose(candidate.dt, model.dt) for candidate in additional_models):
        raise ValueError("models loaded together must use the same dt")
    event_sources, signal_sources, trial_value_sources = _required_sources(models)
    _validate_sources(event_sources, signal_sources, trial_value_sources)

    trials = lazynwb.read_nwb(nwb_path, "/intervals/trials")
    starts = trials.select("start_time").to_numpy().ravel()
    ends = trials.select("stop_time").to_numpy().ravel()
    task_start = float(starts[0])
    task_end = float(ends[-1])
    n_time = int(np.floor((task_end - task_start) / model.dt))

    start_relative = starts - task_start
    duration = task_end - task_start
    trial_ends = np.append(start_relative[1:], duration)
    trial_index = make_trial_index(start_relative, trial_ends, model.dt, n_time=n_time)

    events = {
        source: _trial_events(trials, task_start, source)
        for source in sorted(event_sources & _TRIAL_EVENT_SOURCES)
    }
    for source, path in (
        ("licks", "/processing/behavior/licks"),
        ("rewards", "/processing/behavior/rewards"),
    ):
        if source in event_sources:
            times = (
                lazynwb.scan_nwb(nwb_path, path)
                .select("timestamps")
                .collect()
                .to_numpy()
                .ravel()
            )
            events[source] = times - task_start

    signals: dict[str, TimedSignal] = {}
    if "running_speed" in signal_sources:
        running = (
            lazynwb.scan_nwb(nwb_path, "/processing/behavior/running_speed")
            .select("timestamps", "data")
            .collect()
            .to_numpy()
        )
        signals["running_speed"] = TimedSignal(
            running[:, 1], running[:, 0] - task_start
        )
    if "pupil_area" in signal_sources:
        pupil = (
            lazynwb.scan_nwb(nwb_path, "/processing/behavior/eye_tracking")
            .filter(~pl.col("pupil_is_bad_frame"))
            .select("timestamps", "pupil_area")
            .collect()
            .to_numpy()
        )
        signals["pupil_area"] = TimedSignal(pupil[:, 1], pupil[:, 0] - task_start)

    requested_face_sources = [
        source for source in _FACE_SIGNAL_FEATURES if source in signal_sources
    ]
    if requested_face_sources:
        pose_columns = [
            feature + suffix
            for source in requested_face_sources
            for feature in (_FACE_SIGNAL_FEATURES[source],)
            for suffix in ("_x", "_y", "_likelihood", "_temporal_norm")
        ]
        pose = (
            lazynwb.scan_nwb(nwb_path, "/processing/behavior/lp_side_camera")
            .select(*pose_columns)
            .collect()
        )
        side_frame_times = (
            lazynwb.scan_nwb(nwb_path, "/acquisition/frametimes_side_camera")
            .select("timestamps")
            .collect()
            .to_numpy()
            .ravel()
        )
        for source in requested_face_sources:
            signals[source] = _pose_signal(
                pose,
                side_frame_times,
                task_start,
                _FACE_SIGNAL_FEATURES[source],
            )

    trial_values = {}
    if "context_baseline" in signal_sources or "trial_context" in trial_value_sources:
        trial_context = trials["is_vis_rewarded"].to_numpy().astype(int) * 2 - 1
        if "context_baseline" in signal_sources:
            # Hold each trial's label constant over all of its bins, with an
            # instantaneous step at the trial boundary.
            signals["context_baseline"] = TimedSignal(trial_context[trial_index])
        if "trial_context" in trial_value_sources:
            trial_values["trial_context"] = trial_context
    if "time" in signal_sources:
        signals["time"] = TimedSignal(np.arange(n_time) / n_time)

    data = ModelData(
        dt=model.dt,
        trial_index=trial_index,
        events=events,
        signals=signals,
        trial_values=trial_values,
    )
    return SessionData(
        nwb_path=nwb_path,
        task_start_time=task_start,
        task_end_time=task_end,
        data=data,
    )


def stimulus_mask(
    data: ModelData,
    window: tuple[float, float] = STIMULUS_FIT_WINDOW,
) -> np.ndarray:
    stimulus_times = np.concatenate([data.events[name] for name in STIMULUS_EVENTS])
    return windows_mask(stimulus_times, data.n_time, data.dt, window)


def prepare(
    session: SessionData,
    model: ModelSpec,
) -> PreparedDesign:
    """Build one session-shared design for a declared model."""
    return compile_design(model, session.data)


def qc_unit_ids(nwb_path: str, *, qc_column: str = QC_COLUMN) -> list[str]:
    return (
        lazynwb.scan_nwb(nwb_path, "/units")
        .filter(pl.col(qc_column) & (pl.col("decoder_label") != "noise"))
        .select("unit_id")
        .collect()["unit_id"]
        .to_list()
    )


def load_unit_target(session: SessionData, unit_id: str) -> np.ndarray:
    spikes = (
        lazynwb.scan_nwb(session.nwb_path, "/units")
        .filter(pl.col("unit_id") == unit_id)
        .select("spike_times")
        .collect()["spike_times"]
    )
    if spikes.is_empty():
        raise ValueError(f"unit {unit_id!r} was not found in {session.nwb_path}")
    return bin_spike_times(
        spikes[0].to_numpy(),
        session.task_start_time,
        session.task_end_time,
        session.dt,
    )

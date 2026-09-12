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
    _window_bin_bounds,
)
from .design import PreparedDesign, _bin_rows, compile_design
from .model import Dropout, Event, Gain, ModelSpec, Signal

lazynwb.config.anon = True

DEFAULT_DT = 0.025
STIMULUS_FIT_WINDOW = (-1.0, 1.0)
QC_COLUMN = "default_qc"
INSTRUCTION_TRIAL_COLUMN = "is_instruction"
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
        window=(-0.1, 0.1),
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
        window=(-0.5, 0.5),
        n_basis=5,
        normalize="zscore",
        groups=("behavior",),
        orthogonalize_against="context_baseline",
    ),
    Signal(
        "pupil_area",
        window=(-0.5, 0.5),
        n_basis=5,
        normalize="zscore",
        groups=("behavior",),
        orthogonalize_against="context_baseline",
    ),
    Signal(
        "ear",
        window=(-0.2, 0.2),
        n_basis=4,
        normalize="zscore",
        groups=("behavior", "face"),
        orthogonalize_against="context_baseline",
    ),
    Signal(
        "jaw",
        window=(-0.2, 0.2),
        n_basis=4,
        normalize="zscore",
        groups=("behavior", "face"),
        orthogonalize_against="context_baseline",
    ),
    Signal(
        "nose",
        window=(-0.2, 0.2),
        n_basis=4,
        normalize="zscore",
        groups=("behavior", "face"),
        orthogonalize_against="context_baseline",
    ),
    Signal(
        "whisker_pad",
        window=(-0.2, 0.2),
        n_basis=4,
        normalize="zscore",
        groups=("behavior", "face"),
        orthogonalize_against="context_baseline",
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
    # Dropout.gain("context"),
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
    # Dropout.gain_terms(
    #     "context",
    #     "rewards",
    #     name="reward_context_gain",
    # ),
    # Dropout.gain_terms(
    #     "context",
    #     "licks",
    #     name="lick_context_gain",
    # ),
    Dropout.predictors("context_baseline"),
)

DEFAULT_MODEL = ModelSpec(
    predictors=(
        *_stimulus_predictors(),
        *_late_stimulus_predictors(),
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
    """Session bounds and stimulus-aligned Dynamic Routing model rows."""

    nwb_path: str
    task_start_time: float
    task_end_time: float
    data: ModelData
    included_trial_mask: np.ndarray
    bin_starts: np.ndarray

    def __post_init__(self) -> None:
        included = np.asarray(self.included_trial_mask, dtype=bool).ravel().copy()
        if included.size != self.data.n_trials:
            raise ValueError(
                "included_trial_mask must have one value per indexed trial"
            )
        if not included.any():
            raise ValueError("included_trial_mask must include at least one trial")
        bin_starts = np.asarray(self.bin_starts, dtype=float).ravel().copy()
        if bin_starts.size != self.data.n_time:
            raise ValueError("bin_starts must have one value per model row")
        if not np.all(np.isfinite(bin_starts)) or np.any(np.diff(bin_starts) <= 0):
            raise ValueError("bin_starts must be finite and strictly increasing")
        if bin_starts[0] < self.task_start_time - 1e-9 or (
            bin_starts[-1] + self.data.dt > self.task_end_time + 1e-9
        ):
            raise ValueError("bin_starts must lie within the task interval")
        included.setflags(write=False)
        bin_starts.setflags(write=False)
        object.__setattr__(self, "included_trial_mask", included)
        object.__setattr__(self, "bin_starts", bin_starts)

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


def _filter_events_to_included_trials(
    times: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    included_trials: np.ndarray,
) -> np.ndarray:
    """Remove global events occurring during excluded trial intervals."""
    values = np.asarray(times, dtype=float).ravel()
    trial_index = np.searchsorted(starts, values, side="right") - 1
    in_trial = (trial_index >= 0) & (trial_index < included_trials.size)
    valid_index = np.flatnonzero(in_trial)
    keep = np.ones(values.size, dtype=bool)
    keep[valid_index] = included_trials[trial_index[valid_index]]
    return values[keep]


def _context_block_index(context: np.ndarray) -> np.ndarray:
    """Number contiguous runs of one rewarded context in trial order."""
    values = np.asarray(context).ravel()
    if values.size == 0:
        raise ValueError("cannot identify context blocks without trials")
    blocks = np.zeros(values.size, dtype=int)
    blocks[1:] = np.cumsum(values[1:] != values[:-1])
    return blocks


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


def _snap_grid_coordinates(values: np.ndarray) -> np.ndarray:
    """Snap floating-point values that are effectively integer bin offsets."""
    nearest = np.rint(values)
    return np.where(np.isclose(values, nearest, rtol=0, atol=1e-9), nearest, values)


def _stimulus_aligned_grid(
    starts: np.ndarray,
    ends: np.ndarray,
    stimulus_times: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complete per-trial bins whose zero edge is stimulus onset."""
    starts = np.asarray(starts, dtype=float).ravel()
    ends = np.asarray(ends, dtype=float).ravel()
    stimulus_times = np.asarray(stimulus_times, dtype=float).ravel()
    if not (starts.size == ends.size == stimulus_times.size) or starts.size == 0:
        raise ValueError("trials must provide equally sized starts, ends, and stimuli")
    if not (
        np.all(np.isfinite(starts))
        and np.all(np.isfinite(ends))
        and np.all(np.isfinite(stimulus_times))
    ):
        raise ValueError("trial times and stimulus times must be finite")
    if np.any(ends <= starts):
        raise ValueError("each trial end must exceed its start")
    if np.any(stimulus_times < starts) or np.any(stimulus_times >= ends):
        raise ValueError("each stimulus onset must lie within its trial")

    first_lags = np.ceil(_snap_grid_coordinates((starts - stimulus_times) / dt)).astype(
        int
    )
    end_lags = np.floor(_snap_grid_coordinates((ends - stimulus_times) / dt)).astype(
        int
    )
    counts = end_lags - first_lags
    if np.any(counts <= 0):
        raise ValueError("each trial must contain at least one complete aligned bin")

    bin_starts = np.concatenate(
        [
            stimulus + np.arange(first, end) * dt
            for stimulus, first, end in zip(stimulus_times, first_lags, end_lags)
        ]
    )
    trial_index = np.repeat(np.arange(starts.size), counts)
    return bin_starts, trial_index


def _event_windows_mask(
    session: SessionData,
    event_sources: tuple[str, ...],
    window: tuple[float, float],
) -> np.ndarray:
    """Select event windows without allowing them to cross trial boundaries."""
    missing = set(event_sources) - set(session.data.events)
    if missing:
        raise KeyError(f"fit event sources are missing: {sorted(missing)}")
    event_times = np.concatenate(
        [session.data.events[source] for source in event_sources]
    )
    if not np.all(np.isfinite(event_times)):
        raise ValueError("fit event sources contain non-finite times")

    relative_bin_starts = session.bin_starts - session.task_start_time
    event_rows = _bin_rows(
        event_times,
        session.dt,
        session.n_time,
        relative_bin_starts,
    )
    low, high = _window_bin_bounds(window, session.dt)
    mask = np.zeros(session.n_time, dtype=bool)
    trial_index = session.data.trial_index
    for event_row in event_rows:
        trial = trial_index[event_row]
        trial_start = int(np.searchsorted(trial_index, trial, side="left"))
        trial_end = int(np.searchsorted(trial_index, trial, side="right"))
        first = max(trial_start, event_row + low)
        last = min(trial_end, event_row + high)
        mask[first:last] = True
    return mask


def load_session(
    nwb_path: str,
    model: ModelSpec,
    *additional_models: ModelSpec,
    use_instruction_trials: bool = False,
) -> SessionData:
    """Load the union of session inputs required by the supplied models."""
    models = (model, *additional_models)
    if any(not np.isclose(candidate.dt, model.dt) for candidate in additional_models):
        raise ValueError("models loaded together must use the same dt")
    event_sources, signal_sources, trial_value_sources = _required_sources(models)
    _validate_sources(event_sources, signal_sources, trial_value_sources)

    trials = lazynwb.read_nwb(nwb_path, "/intervals/trials")
    if INSTRUCTION_TRIAL_COLUMN not in trials.columns:
        raise KeyError(
            f"trial table is missing required column {INSTRUCTION_TRIAL_COLUMN!r}"
        )
    instruction_trials = (
        trials[INSTRUCTION_TRIAL_COLUMN].fill_null(False).to_numpy().astype(bool)
    )
    included_trial_mask = (
        np.ones(instruction_trials.size, dtype=bool)
        if use_instruction_trials
        else ~instruction_trials
    )
    if not included_trial_mask.any():
        raise ValueError("no non-instruction trials are available")
    starts = trials.select("start_time").to_numpy().ravel()
    ends = trials.select("stop_time").to_numpy().ravel()
    stimulus_times = trials.select("stim_start_time").to_numpy().ravel()
    task_start = float(starts[0])
    task_end = float(ends[-1])
    aligned_trial_ends = np.append(starts[1:], task_end)
    bin_starts, trial_index = _stimulus_aligned_grid(
        starts, aligned_trial_ends, stimulus_times, model.dt
    )

    start_relative = starts - task_start
    trial_ends = aligned_trial_ends - task_start
    event_trials = trials.filter(pl.Series("included", included_trial_mask))

    events = {
        source: _trial_events(event_trials, task_start, source)
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
            events[source] = _filter_events_to_included_trials(
                times - task_start,
                start_relative,
                trial_ends,
                included_trial_mask,
            )

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
    trial_context = trials["is_vis_rewarded"].to_numpy().astype(int) * 2 - 1
    if "context_baseline" in signal_sources or "trial_context" in trial_value_sources:
        if "context_baseline" in signal_sources:
            # Hold each trial's label constant over all of its bins, with an
            # instantaneous step at the trial boundary.
            signals["context_baseline"] = TimedSignal(trial_context[trial_index])
        if "trial_context" in trial_value_sources:
            trial_values["trial_context"] = trial_context
    if "time" in signal_sources:
        signals["time"] = TimedSignal(
            (bin_starts + model.dt / 2 - task_start) / (task_end - task_start)
        )

    data = ModelData(
        dt=model.dt,
        trial_index=trial_index,
        events=events,
        signals=signals,
        trial_values=trial_values,
        cv_groups=_context_block_index(trial_context),
    )
    return SessionData(
        nwb_path=nwb_path,
        task_start_time=task_start,
        task_end_time=task_end,
        data=data,
        included_trial_mask=included_trial_mask,
        bin_starts=bin_starts,
    )


def stimulus_mask(
    session: SessionData,
    window: tuple[float, float] = STIMULUS_FIT_WINDOW,
) -> np.ndarray:
    return _event_windows_mask(session, STIMULUS_EVENTS, window)


def prepare(
    session: SessionData,
    model: ModelSpec,
) -> PreparedDesign:
    """Build one trial-segmented design on the stimulus-aligned grid."""
    included_rows = session.included_trial_mask[session.data.trial_index]
    fit_mask = (
        None
        if model.fit_window is None
        else _event_windows_mask(session, model.fit_events, model.fit_window)
    )
    return compile_design(
        model,
        session.data,
        fit_mask=fit_mask,
        row_mask=included_rows,
        _bin_starts=session.bin_starts - session.task_start_time,
        _convolution_segments=session.data.trial_index,
    )


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
    rows = _bin_rows(
        spikes[0].to_numpy(),
        session.dt,
        session.n_time,
        session.bin_starts,
    )
    target = np.zeros(session.n_time)
    np.add.at(target, rows, 1.0)
    return target

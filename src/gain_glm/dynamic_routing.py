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
from .evaluation import Dropout
from .model import Event, Gain, ModelSpec, Signal

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
    *, window: tuple[float, float] = (0, 0.1), n_basis: int = 2
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

DEFAULT_MODEL = ModelSpec(
    predictors=(
        *_stimulus_predictors(),
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
)

NO_FACE_MODEL = DEFAULT_MODEL.without_group("face", name="no_face")

NO_HIT_LONG_STIM_MODEL = ModelSpec(
    predictors=(*_stimulus_predictors(window=(0, 1), n_basis=10), *BEHAVIOR_PREDICTORS),
    gains=DEFAULT_MODEL.gains,
    name="no_hit_long_stim",
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

MODELS: Mapping[str, ModelSpec] = {
    model.name: model
    for model in (
        DEFAULT_MODEL,
        NO_FACE_MODEL,
        NO_HIT_LONG_STIM_MODEL,
        ALL_RESPONSE_MODEL,
    )
}
DEFAULT_DROPOUTS = (Dropout.gain("context"),
                    Dropout.predictors('context_baseline'))


@dataclass(frozen=True)
class SessionData:
    nwb_path: str
    dt: float
    task_start_time: float
    task_end_time: float
    n_time: int
    trials: pl.DataFrame
    trial_start_times: np.ndarray
    trial_end_times: np.ndarray
    trial_context: np.ndarray
    lick_times: np.ndarray
    reward_times: np.ndarray
    running_speed: np.ndarray
    pupil: np.ndarray
    pose: pl.DataFrame
    side_frame_times: np.ndarray


def load_session(nwb_path: str, *, dt: float = DEFAULT_DT) -> SessionData:
    """Read each session-level NWB stream once."""
    trials = lazynwb.read_nwb(nwb_path, "/intervals/trials")
    starts = trials.select("start_time").to_numpy().ravel()
    ends = trials.select("stop_time").to_numpy().ravel()
    task_start = float(starts[0])
    task_end = float(ends[-1])
    n_time = int(np.floor((task_end - task_start) / dt))

    licks = (
        lazynwb.scan_nwb(nwb_path, "/processing/behavior/licks")
        .select("timestamps")
        .collect()
        .to_numpy()
        .ravel()
    )
    rewards = (
        lazynwb.scan_nwb(nwb_path, "/processing/behavior/rewards")
        .select("timestamps")
        .collect()
        .to_numpy()
        .ravel()
    )
    running = (
        lazynwb.scan_nwb(nwb_path, "/processing/behavior/running_speed")
        .select("timestamps", "data")
        .collect()
        .to_numpy()
    )
    pupil = (
        lazynwb.scan_nwb(nwb_path, "/processing/behavior/eye_tracking")
        .filter(~pl.col("pupil_is_bad_frame"))
        .select("timestamps", "pupil_area")
        .collect()
        .to_numpy()
    )
    pose = lazynwb.scan_nwb(nwb_path, "/processing/behavior/lp_side_camera").collect()
    side_times = (
        lazynwb.scan_nwb(nwb_path, "/acquisition/frametimes_side_camera")
        .select("timestamps")
        .collect()
        .to_numpy()
        .ravel()
    )
    context = trials["is_vis_rewarded"].to_numpy().astype(int) * 2 - 1
    return SessionData(
        nwb_path=nwb_path,
        dt=dt,
        task_start_time=task_start,
        task_end_time=task_end,
        n_time=n_time,
        trials=trials,
        trial_start_times=starts,
        trial_end_times=ends,
        trial_context=context,
        lick_times=licks,
        reward_times=rewards,
        running_speed=running,
        pupil=pupil,
        pose=pose,
        side_frame_times=side_times,
    )


def _trial_events(session: SessionData, column: str) -> np.ndarray:
    if column not in session.trials.columns:
        return np.zeros(0)
    return (
        session.trials.filter(pl.col(column).fill_null(False))
        .select("stim_start_time")["stim_start_time"]
        .drop_nulls()
        .to_numpy()
        - session.task_start_time
    )


def _pose_signal(
    session: SessionData,
    feature: str,
    *,
    likelihood_min: float = 0.98,
    jitter_sd: float = 3.0,
) -> TimedSignal:
    x = session.pose.select(feature + "_x").to_numpy().ravel()
    y = session.pose.select(feature + "_y").to_numpy().ravel()
    likelihood = session.pose.select(feature + "_likelihood").to_numpy().ravel()
    temporal_norm = session.pose.select(feature + "_temporal_norm").to_numpy().ravel()
    valid = (likelihood > likelihood_min) & (
        temporal_norm
        <= np.nanmean(temporal_norm) + jitter_sd * np.nanstd(temporal_norm)
    )
    return TimedSignal(
        np.sqrt(x[valid] ** 2 + y[valid] ** 2),
        session.side_frame_times[valid] - session.task_start_time,
    )


def model_data(session: SessionData) -> ModelData:
    """Resolve the experiment's NWB schema into named, model-ready inputs."""
    event_names = (*STIMULUS_EVENTS, *OUTCOME_EVENTS)
    events = {name: _trial_events(session, name) for name in event_names}
    events.update(
        {
            "licks": session.lick_times - session.task_start_time,
            "rewards": session.reward_times - session.task_start_time,
        }
    )

    start_relative = session.trial_start_times - session.task_start_time
    duration = session.task_end_time - session.task_start_time
    trial_ends = np.append(start_relative[1:], duration)
    trial_index = make_trial_index(
        start_relative, trial_ends, session.dt, n_time=session.n_time
    )
    signals = {
        "running_speed": TimedSignal(
            session.running_speed[:, 1],
            session.running_speed[:, 0] - session.task_start_time,
        ),
        "pupil_area": TimedSignal(
            session.pupil[:, 1],
            session.pupil[:, 0] - session.task_start_time,
        ),
        "ear": _pose_signal(session, "ear_base_l"),
        "jaw": _pose_signal(session, "jaw"),
        "nose": _pose_signal(session, "nose_tip"),
        "whisker_pad": _pose_signal(session, "whisker_pad_l_side"),
        # Additive context baseline: hold each trial's label constant over all
        # of its bins, with an instantaneous step at the trial boundary.
        "context_baseline": TimedSignal(session.trial_context[trial_index]),
        "time": TimedSignal(np.arange(session.n_time) / session.n_time),
    }
    return ModelData(
        dt=session.dt,
        trial_index=trial_index,
        events=events,
        signals=signals,
        trial_values={"trial_context": session.trial_context},
    )


def stimulus_mask(
    data: ModelData,
    window: tuple[float, float] = STIMULUS_FIT_WINDOW,
) -> np.ndarray:
    stimulus_times = np.concatenate([data.events[name] for name in STIMULUS_EVENTS])
    return windows_mask(stimulus_times, data.n_time, data.dt, window)


def prepare(
    session: SessionData,
    model: ModelSpec = DEFAULT_MODEL,
    *,
    fit_window: tuple[float, float] | None = STIMULUS_FIT_WINDOW,
) -> PreparedDesign:
    """Build one session-shared design for a declared model."""
    data = model_data(session)
    mask = None if fit_window is None else stimulus_mask(data, fit_window)
    return compile_design(model, data, fit_mask=mask)


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

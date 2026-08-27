"""Dynamic Routing NWB adapter, default models, and batch fitting.

Everything specific to the experiment's NWB schema lives here. The core GLM
only sees a :class:`ModelData` object containing named arrays.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import lazynwb
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from .data import ModelData, TimedSignal, bin_spike_times, make_trial_index, windows_mask
from .design import PreparedDesign, compile_design
from .evaluation import CVConfig, Dropout
from .model import Event, FitConfig, Gain, ModelSpec, Signal

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
        groups=("context", "nuisance"),
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

NO_FACE_MODEL = ModelSpec(
    DEFAULT_MODEL.without_group("face").predictors,
    DEFAULT_MODEL.gains,
    name="no_face",
)

NO_HIT_LONG_STIM_MODEL = ModelSpec(
    predictors=(*_stimulus_predictors(window=(0, 1), n_basis=10), *BEHAVIOR_PREDICTORS),
    gains=DEFAULT_MODEL.gains,
    name="no_hit_long_stim",
)

ALL_RESPONSE_MODEL = ModelSpec(
    predictors=(
        *DEFAULT_MODEL.predictors,
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
    ),
    gains=DEFAULT_MODEL.gains,
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
DEFAULT_DROPOUTS = (Dropout.gain("context"),)


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
    pose = lazynwb.scan_nwb(
        nwb_path, "/processing/behavior/lp_side_camera"
    ).collect()
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
        "context_baseline": TimedSignal(session.trial_context, start_relative),
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
    stimulus_times = np.concatenate(
        [data.events[name] for name in STIMULUS_EVENTS]
    )
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


def qc_unit_ids(
    nwb_path: str, *, qc_column: str = QC_COLUMN
) -> list[str]:
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


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _evaluate_unit(
    unit_id: str,
    y: np.ndarray,
    prepared: PreparedDesign,
    fit: FitConfig,
    cv: CVConfig,
    dropouts: Sequence[Dropout],
) -> tuple[str, dict]:
    with threadpool_limits(1):
        result = prepared.evaluate(y, fit=fit, cv=cv, dropouts=dropouts)
    return unit_id, result.to_dict()


def fit_session(
    nwb_path: str,
    session_id: str,
    output_dir: str | Path,
    *,
    model: ModelSpec = DEFAULT_MODEL,
    dropouts: Sequence[Dropout] = DEFAULT_DROPOUTS,
    fit: FitConfig | None = None,
    cv: CVConfig | None = None,
    dt: float = DEFAULT_DT,
    fit_window: tuple[float, float] | None = STIMULUS_FIT_WINDOW,
    n_jobs: int = -1,
    limit: int | None = None,
    qc_column: str = QC_COLUMN,
    overwrite: bool = False,
) -> Path:
    """Fit every selected unit while reusing one prepared session design."""
    output_path = Path(output_dir) / f"{session_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        print(f"[{session_id}] {output_path} exists; skipping")
        return output_path

    units = qc_unit_ids(nwb_path, qc_column=qc_column)
    if limit is not None:
        units = units[:limit]
    session = load_session(nwb_path, dt=dt)
    prepared = prepare(session, model, fit_window=fit_window)
    targets = {unit: load_unit_target(session, unit) for unit in units}
    fit_config = FitConfig(max_iter=50) if fit is None else fit
    cv_config = CVConfig() if cv is None else cv
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_evaluate_unit)(
            unit, targets[unit], prepared, fit_config, cv_config, tuple(dropouts)
        )
        for unit in units
    )
    payload = {
        "session_id": session_id,
        "model": model.name,
        "dt": dt,
        "units": dict(results),
    }
    with output_path.open("w") as stream:
        json.dump(payload, stream, default=_json_default)
    return output_path


def compare_models(
    nwb_path: str,
    models: Sequence[ModelSpec],
    *,
    unit_ids: Sequence[str] | None = None,
    dropouts: Sequence[Dropout] = DEFAULT_DROPOUTS,
    fit: FitConfig | None = None,
    cv: CVConfig | None = None,
    dt: float = DEFAULT_DT,
    fit_window: tuple[float, float] | None = STIMULUS_FIT_WINDOW,
    unit_limit: int = 8,
) -> pl.DataFrame:
    """Compare full-model variants using one session load and target cache."""
    if not models:
        raise ValueError("at least one model is required")
    session = load_session(nwb_path, dt=dt)
    inputs = model_data(session)
    mask = None if fit_window is None else stimulus_mask(inputs, fit_window)
    prepared = {
        model.name: compile_design(model, inputs, fit_mask=mask) for model in models
    }
    if len(prepared) != len(models):
        raise ValueError("model names must be unique")
    units = (
        list(unit_ids)
        if unit_ids is not None
        else qc_unit_ids(nwb_path)[:unit_limit]
    )
    targets = {unit: load_unit_target(session, unit) for unit in units}
    fit_config = FitConfig(max_iter=50) if fit is None else fit
    cv_config = CVConfig() if cv is None else cv

    rows: list[dict[str, Any]] = []
    for model in models:
        for unit in units:
            with threadpool_limits(1):
                result = prepared[model.name].evaluate(
                    targets[unit],
                    fit=fit_config,
                    cv=cv_config,
                    dropouts=dropouts,
                )
            row: dict[str, Any] = {
                "model": model.name,
                "unit_id": unit,
                "train_r2": result.train_r2,
                "cv_r2": result.cv.r2,
                "worst_fold_r2": float(np.min(result.cv.r2_per_fold)),
                "n_iter": result.fit.n_iter,
                "converged": result.fit.converged,
            }
            for name, dropout in result.dropouts.items():
                row[f"delta_r2_{name}"] = dropout.delta_r2
            rows.append(row)
    return pl.DataFrame(rows)


def parse_dropout(value: str) -> Dropout:
    """Parse ``gain:name``, ``group:name``, or ``predictors:a,b``."""
    try:
        kind, names = value.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "dropouts use gain:name, group:name, or predictors:a,b"
        ) from error
    if kind == "gain":
        return Dropout.gain(names)
    if kind == "group":
        return Dropout.group(names)
    if kind == "predictors":
        predictors = tuple(name for name in names.split(",") if name)
        return Dropout.predictors(*predictors)
    raise argparse.ArgumentTypeError(f"unknown dropout kind {kind!r}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nwb-path", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=MODELS, default="default")
    parser.add_argument("--dropout", action="append", type=parse_dropout)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--qc-column", default=QC_COLUMN)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    fit_session(
        args.nwb_path,
        args.session_id,
        args.output_dir,
        model=MODELS[args.model],
        dropouts=args.dropout or DEFAULT_DROPOUTS,
        fit=FitConfig(max_iter=args.max_iter),
        cv=CVConfig(folds=args.folds, seed=args.fold_seed),
        dt=args.dt,
        n_jobs=args.n_jobs,
        limit=args.limit,
        qc_column=args.qc_column,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

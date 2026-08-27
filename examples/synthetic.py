"""Minimal end-to-end example with two gains and two reduced models."""

import numpy as np

from gain_glm import (
    CVConfig,
    Dropout,
    Event,
    FitConfig,
    Gain,
    ModelData,
    ModelSpec,
    Signal,
    compile_design,
)


def main() -> None:
    rng = np.random.default_rng(4)
    n_trials = 40
    bins_per_trial = 30
    n_time = n_trials * bins_per_trial
    trial_index = np.repeat(np.arange(n_trials), bins_per_trial)
    context = np.tile([-1.0, 1.0], n_trials // 2)

    cue_times = np.arange(n_trials) * bins_per_trial * 0.02 + 0.1
    running = rng.normal(size=n_time)
    cue_series = np.zeros(n_time)
    cue_series[np.floor(cue_times / 0.02).astype(int)] = 1
    cue_drive = np.convolve(cue_series, np.exp(-np.arange(20) / 5), mode="full")[:n_time]
    y = (
        0.2
        + (1.5 + 0.8 * context[trial_index]) * cue_drive
        + 0.25 * running
        + rng.normal(0, 0.12, n_time)
    )

    data = ModelData(
        dt=0.02,
        trial_index=trial_index,
        events={"cue": cue_times},
        signals={"running": running},
        trial_values={"context": context},
    )
    model = ModelSpec(
        predictors=(
            Event(
                "cue",
                window=(0, 0.38),
                n_basis=8,
                gains=("context",),
                groups=("task",),
            ),
            Signal(
                "running",
                window=(0, 0),
                n_basis=1,
                groups=("behavior",),
            ),
        ),
        gains=(Gain("context"),),
        name="cue_gain",
    )

    prepared = compile_design(model, data)
    result = prepared.evaluate(
        y,
        fit=FitConfig(regularizer="ridge"),
        cv=CVConfig(folds=5, seed=0),
        dropouts=[
            Dropout.gain("context"),
            Dropout.group("behavior"),
        ],
    )

    print(f"train R²: {result.train_r2:.3f}")
    print(f"CV R²:    {result.cv.r2:.3f}")
    for name, dropout in result.dropouts.items():
        print(
            f"drop {name:>8}: ΔR²={dropout.delta_r2:.3f} "
            f"({dropout.dropout.refit} refit)"
        )
    print(result.fit.gain_table())


if __name__ == "__main__":
    main()

"""Recover a known trial-by-trial context gain from synthetic responses.

The synthetic cue kernel has unit L2 norm, matching the fitted model's gauge.
Consequently the scalar multiplying that kernel is the gain itself. We also
estimate an empirical gain on every trial by projecting the noisy cue response
onto the known kernel, then compare its context regression with the GLM fit.
"""

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
    dt = 0.02
    n_trials = 40
    bins_per_trial = 30
    n_time = n_trials * bins_per_trial
    trial_index = np.repeat(np.arange(n_trials), bins_per_trial)
    context = np.tile([-1.0, 1.0], n_trials // 2)

    # The quantity we want the model to recover:
    #
    #     gain(trial) = 1.5 + 0.8 * context(trial)
    #
    # Thus auditory-context trials (context=-1) have gain 0.7 and visual-
    # context trials (context=+1) have gain 2.3.
    true_gain_offset = 1.5
    true_gain_context = 0.8
    true_gain_by_trial = true_gain_offset + true_gain_context * context

    # Put one cue five bins into every trial. Its response shape is a decaying
    # 20-bin kernel. Unit normalization makes its amplitude directly equal to
    # true_gain_by_trial, with no scale ambiguity between kernel and gain.
    cue_offset_bins = 5
    kernel_bins = 20
    cue_bins = np.arange(n_trials) * bins_per_trial + cue_offset_bins
    cue_times = cue_bins * dt
    true_kernel = np.exp(-np.arange(kernel_bins) / 5)
    true_kernel /= np.linalg.norm(true_kernel)

    running = rng.normal(size=n_time)
    cue_series = np.zeros(n_time)
    cue_series[cue_bins] = 1
    cue_drive = np.convolve(cue_series, true_kernel, mode="full")[:n_time]

    true_intercept = 0.2
    true_running_coefficient = 0.25
    y = (
        true_intercept
        + true_gain_by_trial[trial_index] * cue_drive
        + true_running_coefficient * running
        + rng.normal(0, 0.08, n_time)
    )

    # Because this is a simulation, we can remove the known nuisance terms and
    # project each observed cue response onto the known unit-norm kernel. This
    # gives a noisy but direct empirical gain estimate for every trial.
    cue_only = y - true_intercept - true_running_coefficient * running
    empirical_gain_by_trial = np.array(
        [
            cue_only[cue_bin : cue_bin + kernel_bins] @ true_kernel
            for cue_bin in cue_bins
        ]
    )
    empirical_gain_offset, empirical_gain_context = np.linalg.lstsq(
        np.column_stack((np.ones(n_trials), context)),
        empirical_gain_by_trial,
        rcond=None,
    )[0]

    data = ModelData(
        dt=dt,
        trial_index=trial_index,
        events={"cue": cue_times},
        signals={"running": running},
        trial_values={"context": context},
    )
    model = ModelSpec(
        predictors=(
            Event(
                "cue",
                window=(0, kernel_bins * dt),
                n_basis=kernel_bins,
                basis="identity",
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
        dt=dt,
        dropouts=(
            Dropout.gain("context"),
            Dropout.group("behavior"),
        ),
    )

    prepared = compile_design(model, data)
    result = prepared.evaluate(
        y,
        fit=FitConfig(regularizer="ridge", kernel_alpha=1e-3, gain_alpha=1e-3),
        cv=CVConfig(folds=5, seed=0),
    )

    fitted_gain = result.fit.gain_table()["cue"]
    print("Gain model: gain(trial) = offset + context_coef * context(trial)")
    print("                    offset  context_coef")
    print(f"  generating truth   {true_gain_offset:6.3f}      {true_gain_context:6.3f}")
    print(
        f"  empirical response {empirical_gain_offset:6.3f}      "
        f"{empirical_gain_context:6.3f}"
    )
    print(
        f"  fitted GLM         {fitted_gain['offset']:6.3f}      "
        f"{fitted_gain['context']:6.3f}"
    )
    np.testing.assert_allclose(
        [fitted_gain["offset"], fitted_gain["context"]],
        [true_gain_offset, true_gain_context],
        atol=0.1,
    )
    print()
    print(f"train R²: {result.train_r2:.3f}")
    print(f"CV R²:    {result.cv.r2:.3f}")
    for name, dropout in result.dropouts.items():
        print(
            f"drop {name:>8}: ΔR²={dropout.delta_r2:.3f} "
            f"({dropout.dropout.refit} refit)"
        )


if __name__ == "__main__":
    main()

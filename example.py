"""End-to-end demo of bilinear_glm on synthetic data.

Builds a session with two events (cue, action) and one continuous predictor
(running speed). Each event kernel is multiplied trial-by-trial by two latent
"value" variables (V1, V2) plus a baseline gain. The model is asked to recover
the kernels, the per-trial gain offsets, and the value-gain coefficients.
"""
from __future__ import annotations

import numpy as np

from bilinear_glm import (
    BilinearGLM,
    ContinuousPredictor,
    EventPredictor,
    GainModulator,
)


def main():
    rng = np.random.default_rng(0)

    dt = 0.01                            # 10 ms bins
    n_trials = 250
    trial_dur = 3.0                      # seconds
    bins_per_trial = int(trial_dur / dt)
    T = bins_per_trial * n_trials
    trial_idx = np.repeat(np.arange(n_trials), bins_per_trial)

    # ---- event times (one cue & one action per trial) -----------------------
    trial_starts = np.arange(n_trials) * trial_dur
    cue_times = trial_starts + 0.5
    action_times = trial_starts + 1.0 + 0.2 * rng.standard_normal(n_trials)

    # ---- continuous regressor (running speed-like) -------------------------
    running_speed = np.cumsum(rng.standard_normal(T)) * 0.05
    running_speed -= running_speed.mean()
    running_speed /= running_speed.std()

    # ---- ground-truth kernels (unit norm, like the model expects after
    #      normalisation) ----------------------------------------------------
    def make_kernel(window, dt, peak_lag, width):
        lo = int(np.floor(window[0] / dt))
        hi = int(np.ceil(window[1] / dt))
        lags = np.arange(lo, hi + 1) * dt
        k = np.exp(-((lags - peak_lag) ** 2) / (2 * width ** 2))
        k *= np.sign(np.cos((lags - peak_lag) * 2 * np.pi))
        return k / np.linalg.norm(k)

    cue_kernel = make_kernel((0.0, 1.0), dt, peak_lag=0.20, width=0.10)
    action_kernel = make_kernel((-0.5, 0.5), dt, peak_lag=0.05, width=0.08)
    running_kernel = make_kernel((0.0, 0.5), dt, peak_lag=0.15, width=0.08)

    # ---- ground-truth trial-by-trial gains ---------------------------------
    V1 = rng.standard_normal(n_trials)        # e.g. contralateral value
    V2 = rng.standard_normal(n_trials)        # e.g. ipsilateral value

    true_gain = {
        "cue":    {"offset": 1.0, "V1": 0.6,  "V2": 0.0},
        "action": {"offset": 0.8, "V1": -0.4, "V2": 0.5},
    }

    g_cue    = (true_gain["cue"]["offset"]
                + true_gain["cue"]["V1"] * V1
                + true_gain["cue"]["V2"] * V2)
    g_action = (true_gain["action"]["offset"]
                + true_gain["action"]["V1"] * V1
                + true_gain["action"]["V2"] * V2)

    # ---- generate y --------------------------------------------------------
    def conv_event(times, kernel, window):
        bins = np.floor(np.asarray(times) / dt).astype(int)
        s = np.zeros(T)
        bins = bins[(bins >= 0) & (bins < T)]
        np.add.at(s, bins, 1.0)
        lo = int(np.floor(window[0] / dt))
        hi = int(np.ceil(window[1] / dt))
        full = np.convolve(s, kernel, mode="full")
        return full[max(0, -lo): max(0, -lo) + T]

    cue_drive = conv_event(cue_times, cue_kernel, (0.0, 1.0))
    action_drive = conv_event(action_times, action_kernel, (-0.5, 0.5))
    running_drive = np.convolve(running_speed, running_kernel,
                                mode="full")[: T]

    y = (g_cue[trial_idx] * cue_drive
         + g_action[trial_idx] * action_drive
         + running_drive
         + 0.2 * rng.standard_normal(T))

    # ---- specify model -----------------------------------------------------
    predictors = [
        EventPredictor("cue",    cue_times,    window=(0.0, 1.0),
                       n_basis=10, basis="cosine", gain_modulated=True),
        EventPredictor("action", action_times, window=(-0.5, 0.5),
                       n_basis=10, basis="cosine", gain_modulated=True),
        ContinuousPredictor("running_speed", running_speed,
                            window=(0.0, 0.5), n_basis=8,
                            basis="cosine", gain_modulated=False),
    ]
    gains = [
        GainModulator("V1", values=V1, modulates=["cue", "action"]),
        GainModulator("V2", values=V2, modulates=["cue", "action"]),
    ]

    model = BilinearGLM(
        predictors=predictors,
        gains=gains,
        dt=dt,
        kernel_regularizer="ridge",      # ridge is fast for the demo; lasso also works
        alphas=np.logspace(-3, 3, 25),
        # cv_folds=None (default): RidgeCV uses GCV/LOO via SVD — fast
        # cv_folds=5: k-fold CV — needed for LassoCV, slow for Ridge
    )
    model.fit(y=y, trial_idx=trial_idx, max_iter=30, tol=1e-3, verbose=True)

    print("\nin-sample R^2:", model.score(y, trial_idx))

    cv_r2 = model.cross_val_score(y, trial_idx, n_folds=5,
                                   max_iter=30, tol=1e-3, verbose=False)
    print(f"cross-validated R^2 (5-fold trial-held-out): {cv_r2:.4f}")
    print(f"  per-fold: {model.cv_scores_.round(4)}")
    print("\nrecovered gains:")
    for name, row in model.gain_table().items():
        print(f"  {name:8s}", {k: f"{v:+.3f}" for k, v in row.items()})
    print("\ntrue gains:")
    for name, row in true_gain.items():
        print(f"  {name:8s}", {k: f"{v:+.3f}" for k, v in row.items()})

    # kernel correlations
    def correl(a, b):
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b))

    print("\nkernel cosine similarity to ground truth:")
    print(f"  cue           : {correl(model.kernel('cue'), cue_kernel):+.3f}")
    print(f"  action        : {correl(model.kernel('action'), action_kernel):+.3f}")
    print(f"  running_speed : "
          f"{correl(model.kernel('running_speed'), running_kernel):+.3f}")

    # ΔR² comparisons (full vs reduced models)
    print("\nΔR² when each gain is removed:")
    print(f"  remove V1 : {model.delta_r2(y, trial_idx, remove_gains=['V1']):+.5f}")
    print(f"  remove V2 : {model.delta_r2(y, trial_idx, remove_gains=['V2']):+.5f}")
    print(f"  remove V1+V2 : "
          f"{model.delta_r2(y, trial_idx, remove_gains=['V1','V2']):+.5f}")

    print("\n=== Parameter fits ===")
    fig = model.summary_plot(y)
    fig2 = model.plot_fit(y, trial_idx)
    try:
        import matplotlib.pyplot as plt
        fig.savefig("example_summary.png", dpi=120)
        fig2.savefig("example_fit_snippet.png", dpi=120)
        print("\nsaved kernel/PSTH summary to example_summary.png")
        print("saved fit snippet to example_fit_snippet.png")
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()

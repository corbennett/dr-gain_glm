import unittest

import numpy as np

from bilinear_glm import (
    BilinearGLM,
    ContinuousPredictor,
    GainModulator,
)


class _CountingBilinearGLM(BilinearGLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.als_calls = 0

    def _fit_als(self, *args, **kwargs):
        self.als_calls += 1
        return super()._fit_als(*args, **kwargs)


class GainRefitTests(unittest.TestCase):
    def test_reduced_gain_model_refits_retained_coefficients(self):
        rng = np.random.default_rng(2)
        T = 240
        drive = rng.normal(size=T)
        g1 = rng.normal(size=T)
        g2 = 0.9 * g1 + 0.2 * rng.normal(size=T)

        model = BilinearGLM(
            predictors=[
                ContinuousPredictor(
                    "x", drive, window=(0, 0), n_basis=1,
                    basis="identity", gain_modulated=True,
                    outlier_zscore=None,
                )
            ],
            gains=[
                GainModulator("g1", np.zeros(1), modulates=["x"]),
                GainModulator("g2", np.zeros(1), modulates=["x"]),
            ],
            dt=1.0,
            kernel_regularizer="ridge",
            kernel_alpha=1e-8,
            gain_alpha=1e-8,
        )

        drives = {"x": drive}
        gain_t = {"g1": g1, "g2": g2}
        Z = model._build_gain_design(drives, gain_t)
        target = Z @ np.array([1.0, 0.5, 2.0])

        full, _ = model._fit_gain_coefficients(Z, target, alpha=1e-8)
        keep = model._gain_keep_mask(remove_gains=["g2"])
        reduced, _ = model._fit_gain_coefficients(
            Z, target, keep_mask=keep, alpha=1e-8)

        zeroed = full.copy()
        zeroed[model._gain_var_idx[("g2", "x")]] = 0.0
        refit_mse = float(np.mean((target - Z @ reduced) ** 2))
        zeroed_mse = float(np.mean((target - Z @ zeroed) ** 2))

        self.assertLess(refit_mse, zeroed_mse)
        self.assertEqual(reduced[model._gain_var_idx[("g2", "x")]], 0.0)
        self.assertNotAlmostEqual(
            reduced[model._gain_var_idx[("g1", "x")]],
            full[model._gain_var_idx[("g1", "x")]],
        )

    def test_public_evaluators_share_identical_folds_and_results(self):
        rng = np.random.default_rng(3)
        n_trials = 12
        bins_per_trial = 5
        T = n_trials * bins_per_trial
        trial_idx = np.repeat(np.arange(n_trials), bins_per_trial)
        gain_values = np.linspace(-1.0, 1.0, n_trials)
        drive = rng.normal(size=T)
        y = (drive * (2.0 + 1.5 * gain_values[trial_idx])
             + 0.05 * rng.normal(size=T))

        model = BilinearGLM(
            predictors=[
                ContinuousPredictor(
                    "x", drive, window=(0, 0), n_basis=1,
                    basis="identity", gain_modulated=True,
                    outlier_zscore=None,
                )
            ],
            gains=[GainModulator("value", gain_values, modulates=["x"])],
            dt=1.0,
            kernel_regularizer="ridge",
            kernel_alpha=1e-6,
            gain_alpha=1e-6,
        )
        fit_kwargs = dict(
            n_folds=3,
            fold_seed=7,
            max_iter=5,
            tol=1e-8,
            patience=1,
        )

        summary = model.fit_summary(
            y, trial_idx, remove_gains=["value"], **fit_kwargs)
        summary_cv_folds = summary["cv_r2_per_fold"].copy()
        summary_delta_folds = summary["delta_r2_per_fold"].copy()

        cv_r2 = model.cross_val_score(y, trial_idx, **fit_kwargs)
        separate_cv_folds = model.cv_scores_.copy()
        delta_r2 = model.delta_r2(
            y, trial_idx, remove_gains=["value"], **fit_kwargs)

        np.testing.assert_allclose(summary_cv_folds, separate_cv_folds)
        np.testing.assert_allclose(summary_cv_folds, model.cv_scores_)
        np.testing.assert_allclose(
            summary_delta_folds, model.cv_delta_r2_)
        self.assertAlmostEqual(summary["cv_r2"], cv_r2)
        self.assertAlmostEqual(summary["delta_r2"], delta_r2)

    def test_predictor_removal_refits_als_but_gain_removal_does_not(self):
        rng = np.random.default_rng(4)
        n_trials = 9
        bins_per_trial = 4
        trial_idx = np.repeat(np.arange(n_trials), bins_per_trial)
        gain_values = rng.normal(size=n_trials)
        x = rng.normal(size=trial_idx.size)
        nuisance = 0.5 * x + rng.normal(size=trial_idx.size)
        y = (x * (1.0 + gain_values[trial_idx])
             + 0.4 * nuisance
             + 0.05 * rng.normal(size=trial_idx.size))

        model = _CountingBilinearGLM(
            predictors=[
                ContinuousPredictor(
                    "x", x, window=(0, 0), n_basis=1,
                    basis="identity", gain_modulated=True,
                    outlier_zscore=None,
                ),
                ContinuousPredictor(
                    "nuisance", nuisance, window=(0, 0), n_basis=1,
                    basis="identity", gain_modulated=False,
                    outlier_zscore=None,
                ),
            ],
            gains=[GainModulator("value", gain_values, modulates=["x"])],
            dt=1.0,
            kernel_regularizer="ridge",
            kernel_alpha=1e-6,
            gain_alpha=1e-6,
        )
        fit_kwargs = dict(
            n_folds=3,
            fold_seed=5,
            max_iter=2,
            patience=1,
        )

        model.delta_r2(
            y, trial_idx, remove_predictors=["nuisance"], **fit_kwargs)
        self.assertEqual(model.als_calls, 6)

        model.als_calls = 0
        model.delta_r2(
            y, trial_idx, remove_gains=["value"], **fit_kwargs)
        self.assertEqual(model.als_calls, 3)


if __name__ == "__main__":
    unittest.main()

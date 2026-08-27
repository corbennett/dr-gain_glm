import unittest
from unittest import mock

import numpy as np

from gain_glm import (
    CVConfig,
    Dropout,
    FitConfig,
    Gain,
    History,
    ModelData,
    ModelSpec,
    Signal,
    compile_design,
)


def synthetic_problem(seed=0):
    rng = np.random.default_rng(seed)
    n_trials = 18
    bins_per_trial = 16
    trial_index = np.repeat(np.arange(n_trials), bins_per_trial)
    context = np.tile([-1.0, 1.0], n_trials // 2)
    value = context + rng.normal(0, 0.35, n_trials)
    x = rng.normal(size=trial_index.size)
    nuisance = rng.normal(size=trial_index.size)
    y = (
        0.3
        + (1.4 + 0.8 * context[trial_index] + 0.3 * value[trial_index]) * x
        + 0.6 * nuisance
        + rng.normal(0, 0.08, trial_index.size)
    )
    data = ModelData(
        dt=0.05,
        trial_index=trial_index,
        signals={"x": x, "nuisance": nuisance},
        trial_values={"context": context, "value": value},
    )
    spec = ModelSpec(
        predictors=(
            Signal(
                "x",
                window=(0, 0),
                n_basis=1,
                gains=("context", "value"),
                groups=("task",),
            ),
            Signal(
                "nuisance",
                window=(0, 0),
                n_basis=1,
                groups=("behavior",),
            ),
        ),
        gains=(Gain("context"), Gain("value")),
        name="synthetic",
    )
    return compile_design(spec, data), y


class SpecificationTests(unittest.TestCase):
    def test_model_variants_are_immutable(self):
        prepared, _ = synthetic_problem()
        original = prepared.spec
        variant = original.without_group("behavior", name="task_only")
        changed = original.replace("x", window=(-0.1, 0.1), n_basis=3)

        self.assertEqual(original.predictor_names, ("x", "nuisance"))
        self.assertEqual(original.group_members("behavior"), ("nuisance",))
        self.assertEqual(variant.predictor_names, ("x",))
        self.assertEqual(variant.name, "task_only")
        self.assertEqual(changed.predictor("x").window, (-0.1, 0.1))
        self.assertEqual(original.predictor("x").window, (0, 0))

    def test_compile_fails_on_missing_named_source(self):
        data = ModelData(dt=0.1, trial_index=np.repeat(np.arange(2), 5))
        spec = ModelSpec((Signal("missing", window=(0, 0), n_basis=1),))
        with self.assertRaisesRegex(KeyError, "signal source 'missing'"):
            compile_design(spec, data)

    def test_dropout_resolves_groups_and_refit_strategy(self):
        prepared, _ = synthetic_problem()
        gain = Dropout.gain("context").resolve(prepared)
        behavior = Dropout.group("behavior").resolve(prepared)

        self.assertEqual(gain.gains, ("context",))
        self.assertEqual(gain.refit, "gains")
        self.assertEqual(behavior.predictors, ("nuisance",))
        self.assertEqual(behavior.refit, "full")

        mixed = Dropout.terms(
            "task_context", groups=("task",), gains=("context",)
        ).resolve(prepared)
        self.assertEqual(mixed.predictors, ("x",))
        self.assertEqual(mixed.gains, ("context",))
        self.assertEqual(mixed.refit, "full")


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.prepared, self.y = synthetic_problem()
        self.fit = FitConfig(kernel_alpha=1e-6, gain_alpha=1e-6, max_iter=30)

    def test_fit_recovers_parameters(self):
        fitted = self.prepared.fit(self.y, config=self.fit)
        gains = fitted.gain_table()["x"]

        self.assertGreater(fitted.score(self.y), 0.98)
        self.assertAlmostEqual(gains["offset"], 1.4, delta=0.08)
        self.assertAlmostEqual(gains["context"], 0.8, delta=0.08)
        self.assertAlmostEqual(gains["value"], 0.3, delta=0.08)

    def test_multiple_dropouts_share_full_fold_fits(self):
        from gain_glm import evaluation

        original_fit_state = evaluation.fit_state
        original_refit_gains = evaluation.refit_gains
        with (
            mock.patch(
                "gain_glm.evaluation.fit_state", wraps=original_fit_state
            ) as fit_state,
            mock.patch(
                "gain_glm.evaluation.refit_gains", wraps=original_refit_gains
            ) as refit_gains,
        ):
            result = self.prepared.evaluate(
                self.y,
                fit=self.fit,
                cv=CVConfig(folds=3, seed=4),
                dropouts=[
                    Dropout.gain("context"),
                    Dropout.gain("value"),
                    Dropout.group("behavior"),
                ],
            )

        # One full and one predictor-reduced ALS fit per fold. Gain-only
        # comparisons use their own fixed-kernel gain refits instead.
        self.assertEqual(fit_state.call_count, 6)
        self.assertEqual(refit_gains.call_count, 6)
        self.assertEqual(set(result.dropouts), {"context", "value", "behavior"})
        np.testing.assert_equal(
            result.dropouts["context"].delta_r2_per_fold.shape, (3,)
        )

    def test_gain_only_can_request_a_full_refit(self):
        result = self.prepared.evaluate(
            self.y,
            fit=self.fit,
            cv=CVConfig(folds=3),
            dropouts=[Dropout.gain("context", refit="full")],
        )
        self.assertEqual(result.dropouts["context"].dropout.refit, "full")

    def test_evaluation_is_reproducible_and_result_is_typed(self):
        kwargs = {
            "fit": self.fit,
            "cv": CVConfig(folds=3, seed=7),
            "dropouts": [Dropout.gain("context")],
        }
        first = self.prepared.evaluate(self.y, **kwargs)
        second = self.prepared.evaluate(self.y, **kwargs)

        np.testing.assert_allclose(first.cv.r2_per_fold, second.cv.r2_per_fold)
        np.testing.assert_allclose(
            first.dropouts["context"].delta_r2_per_fold,
            second.dropouts["context"].delta_r2_per_fold,
        )
        self.assertIn("context", first.to_dict()["dropouts"])

    def test_history_is_explicit_and_cv_can_gap_boundaries(self):
        rng = np.random.default_rng(12)
        trial_index = np.repeat(np.arange(8), 12)
        x = rng.normal(size=trial_index.size)
        y = rng.normal(size=trial_index.size)
        data = ModelData(dt=0.1, trial_index=trial_index, signals={"x": x})
        spec = ModelSpec(
            (
                Signal("x", window=(0, 0), n_basis=1),
                History(window=(0.1, 0.3), n_basis=3),
            ),
            name="history",
        )
        prepared = compile_design(spec, data)
        result = prepared.evaluate(
            y,
            fit=FitConfig(kernel_alpha=1.0, max_iter=3),
            cv=CVConfig(folds=4, gap_history=True),
        )
        self.assertEqual(result.cv.r2_per_fold.shape, (4,))

    def test_linear_model_converges_in_one_solve(self):
        rng = np.random.default_rng(22)
        trial_index = np.repeat(np.arange(4), 10)
        x = rng.normal(size=trial_index.size)
        data = ModelData(dt=0.1, trial_index=trial_index, signals={"x": x})
        spec = ModelSpec((Signal("x", window=(0, 0), n_basis=1),))
        fitted = compile_design(spec, data).fit(
            0.5 + 2 * x,
            config=FitConfig(kernel_alpha=1e-8, max_iter=1),
        )
        self.assertEqual(fitted.n_iter, 1)
        self.assertTrue(fitted.converged)


if __name__ == "__main__":
    unittest.main()

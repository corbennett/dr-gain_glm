"""Permutation tests against caller-supplied null labels (random block boundaries)."""

import unittest
from unittest import mock

import numpy as np

from gain_glm import (
    CVConfig,
    Dropout,
    FitConfig,
    Gain,
    ModelData,
    ModelSpec,
    NullConfig,
    Signal,
    compile_design,
)
from gain_glm.dynamic_routing import (
    SessionData,
    _random_block_durations,
    random_block_contexts,
    random_block_null,
)

N_BLOCKS = 6
BLOCK_SIZE = 6
N_TRIALS = N_BLOCKS * BLOCK_SIZE


def context_problem(seed=41):
    """Six context blocks with a context gain on ``x`` and a context offset."""
    rng = np.random.default_rng(seed)
    trial_index = np.repeat(np.arange(N_TRIALS), 6)
    context = np.tile(
        np.concatenate([np.full(BLOCK_SIZE, -1.0), np.full(BLOCK_SIZE, 1.0)]),
        N_BLOCKS // 2,
    )
    x = rng.normal(size=trial_index.size)
    target = (
        0.2
        + (1.0 + 0.6 * context[trial_index]) * x
        + 0.5 * context[trial_index]
        + rng.normal(0, 0.3, trial_index.size)
    )
    data = ModelData(
        dt=0.1,
        trial_index=trial_index,
        signals={"x": x, "context_baseline": context[trial_index]},
        trial_values={"context": context},
        cv_groups=np.repeat(np.arange(N_BLOCKS), BLOCK_SIZE),
    )
    spec = ModelSpec(
        predictors=(
            Signal("x", window=(0, 0), n_basis=1, gains=("context",)),
            Signal("context_baseline", window=(0, 0), n_basis=1),
        ),
        gains=(Gain("context"),),
        dt=0.1,
        dropouts=(
            Dropout.gain_terms("context", "x", permutation_test=True),
            Dropout.predictors("context_baseline", permutation_test=True),
        ),
    )
    return compile_design(spec, data), target, context


def random_labels(n_labels, seed=3):
    """Alternating six-block labels with random boundaries, in trial units."""
    rng = np.random.default_rng(seed)
    labels = np.empty((n_labels, N_TRIALS))
    for row in range(n_labels):
        cuts = np.sort(
            rng.choice(np.arange(2, N_TRIALS - 1), N_BLOCKS - 1, replace=False)
        )
        block = np.searchsorted(cuts, np.arange(N_TRIALS), side="right")
        labels[row] = np.where(block % 2 == 0, 1.0, -1.0) * rng.choice([-1.0, 1.0])
    return labels


class LabelNullTests(unittest.TestCase):
    def setUp(self):
        self.prepared, self.target, self.context = context_problem()
        self.labels = random_labels(40)
        self.names = (("context:x", "gain"), ("context_baseline", "predictor"))

    def evaluate(self, fit, labels=None, batched=True):
        labels = self.labels if labels is None else labels
        with mock.patch("gain_glm.evaluation._BATCHED_LABEL_NULL", batched):
            return self.prepared.evaluate(
                self.target,
                fit=fit,
                cv=CVConfig(),
                null=NullConfig(method="labels", labels=labels),
            )

    def test_batched_scores_match_per_label_refits(self):
        for fit in (
            FitConfig(kernel_alpha=1e-3, gain_alpha=1e-3, max_iter=30),
            FitConfig(max_iter=30),  # penalties chosen by inner CV
        ):
            with self.subTest(fit=fit):
                batched = self.evaluate(fit, batched=True)
                looped = self.evaluate(fit, batched=False)
                for name, _ in self.names:
                    a = batched.dropouts[name].permutation
                    b = looped.dropouts[name].permutation
                    np.testing.assert_allclose(
                        a.delta_r2_pooled_per_shift,
                        b.delta_r2_pooled_per_shift,
                        rtol=1e-8,
                        atol=1e-12,
                    )
                    np.testing.assert_allclose(
                        a.delta_r2_per_shift_per_fold,
                        b.delta_r2_per_shift_per_fold,
                        rtol=1e-8,
                        atol=1e-12,
                    )
                    self.assertEqual(a.p_value, b.p_value)

    def test_batched_path_skips_per_label_refits(self):
        from gain_glm import evaluation

        with (
            mock.patch(
                "gain_glm.evaluation._fit_gain_coefficients",
                wraps=evaluation._fit_gain_coefficients,
            ) as fit_gains,
            mock.patch(
                "gain_glm.evaluation.refit_kernels", wraps=evaluation.refit_kernels
            ) as fit_kernels,
        ):
            self.evaluate(FitConfig(kernel_alpha=1e-3, gain_alpha=1e-3, max_iter=30))
        fit_gains.assert_not_called()
        fit_kernels.assert_not_called()

    def test_result_metadata_support_and_p_value(self):
        labels = np.vstack([self.context, self.labels])  # first label = observed
        result = self.evaluate(
            FitConfig(kernel_alpha=1e-3, gain_alpha=1e-3, max_iter=30), labels=labels
        )
        for name, kind in self.names:
            permutation = result.dropouts[name].permutation
            self.assertEqual(permutation.kind, kind)
            self.assertEqual(permutation.method, "labels")
            np.testing.assert_array_equal(
                permutation.shifts, np.arange(labels.shape[0] + 1)
            )
            np.testing.assert_array_equal(permutation.valid_trials, np.arange(N_TRIALS))
            self.assertEqual(permutation.omitted_trials.size, 0)
            curve = permutation.delta_r2_pooled_per_shift
            self.assertEqual(curve.shape, (labels.shape[0] + 1,))
            self.assertEqual(
                permutation.delta_r2_per_shift_per_fold.shape, (curve.size, N_BLOCKS)
            )
            # A label equal to the observed context reproduces the observed score.
            self.assertAlmostEqual(curve[1], curve[0], places=10)
            expected = (1 + np.sum(curve[1:] >= curve[0])) / curve.size
            self.assertAlmostEqual(permutation.p_value, expected)
            self.assertLess(permutation.p_value, 0.1)
        summary = result.to_dict()["dropouts"]["context:x"]["permutation"]
        self.assertEqual(summary["method"], "labels")
        self.assertEqual(summary["n_null"], labels.shape[0])
        self.assertEqual(
            summary["p_value"], result.dropouts["context:x"].permutation.p_value
        )
        self.assertEqual(len(summary["delta_r2_pooled_per_shift"]), labels.shape[0] + 1)
        for key in ("shifts", "delta_r2_per_shift", "delta_r2_per_shift_per_fold"):
            self.assertIsNone(summary[key])

    def test_shift_null_is_unchanged_by_default(self):
        fit = FitConfig(kernel_alpha=1e-3, gain_alpha=1e-3, max_iter=30)
        default = self.prepared.evaluate(self.target, fit=fit, cv=CVConfig())
        explicit = self.prepared.evaluate(
            self.target, fit=fit, cv=CVConfig(), null=NullConfig()
        )
        for name, _ in self.names:
            a = default.dropouts[name].permutation
            b = explicit.dropouts[name].permutation
            self.assertEqual(a.method, "shift")
            np.testing.assert_array_equal(a.shifts, b.shifts)
            np.testing.assert_array_equal(
                a.delta_r2_pooled_per_shift, b.delta_r2_pooled_per_shift
            )

    def test_mapping_labels_are_looked_up_by_gain_or_predictor(self):
        fit = FitConfig(kernel_alpha=1e-3, gain_alpha=1e-3, max_iter=30)
        shared = self.evaluate(fit)
        mapped = self.prepared.evaluate(
            self.target,
            fit=fit,
            cv=CVConfig(),
            null=NullConfig(
                method="labels",
                labels={"context": self.labels, "context_baseline": self.labels},
            ),
        )
        for name, _ in self.names:
            np.testing.assert_allclose(
                shared.dropouts[name].permutation.delta_r2_pooled_per_shift,
                mapped.dropouts[name].permutation.delta_r2_pooled_per_shift,
            )
        with self.assertRaisesRegex(ValueError, "no entry for 'context_baseline'"):
            self.prepared.evaluate(
                self.target,
                fit=fit,
                cv=CVConfig(),
                null=NullConfig(method="labels", labels={"context": self.labels}),
            )

    def test_lasso_kernels_fall_back_to_per_label_refits(self):
        result = self.evaluate(
            FitConfig(
                regularizer="lasso", kernel_alpha=1e-4, gain_alpha=1e-3, max_iter=10
            )
        )
        curve = result.dropouts[
            "context_baseline"
        ].permutation.delta_r2_pooled_per_shift
        self.assertEqual(curve.shape, (self.labels.shape[0] + 1,))
        self.assertTrue(np.all(np.isfinite(curve)))

    def test_invalid_null_configurations(self):
        with self.assertRaisesRegex(ValueError, "null method"):
            NullConfig(method="random")
        with self.assertRaisesRegex(ValueError, "requires labels"):
            NullConfig(method="labels")
        with self.assertRaisesRegex(ValueError, "does not take labels"):
            NullConfig(labels=self.labels)
        with self.assertRaisesRegex(ValueError, "shape"):
            NullConfig(method="labels", labels=np.ones(N_TRIALS))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            NullConfig(method="labels", labels=np.full((2, N_TRIALS), np.nan))
        fit = FitConfig(kernel_alpha=1e-3, gain_alpha=1e-3, max_iter=5)
        with self.assertRaisesRegex(ValueError, "trials, expected"):
            self.evaluate(fit, labels=self.labels[:, :-1])
        with self.assertRaisesRegex(ValueError, "observed predictor values"):
            self.evaluate(fit, labels=0.5 * self.labels)


def synthetic_session(n_trials=120, trial_duration=30.0, n_blocks=6):
    """A one-hour session with one model bin per trial and equal context blocks."""
    starts = np.arange(n_trials) * trial_duration + 0.5
    context = np.repeat(np.tile([1.0, -1.0], n_blocks // 2), n_trials // n_blocks)
    data = ModelData(
        dt=0.025,
        trial_index=np.arange(n_trials),
        signals={},
        trial_values={"trial_context": context},
    )
    return SessionData(
        nwb_path="fake.nwb",
        task_start_time=0.0,
        task_end_time=n_trials * trial_duration,
        data=data,
        included_trial_mask=np.ones(n_trials, dtype=bool),
        bin_starts=starts,
    ), starts


class RandomBlockContextTests(unittest.TestCase):
    def test_durations_respect_minimum_and_total(self):
        durations = _random_block_durations(
            3600.0, 6, 300.0, 500, np.random.default_rng(0)
        )
        self.assertEqual(durations.shape, (500, 6))
        np.testing.assert_allclose(durations.sum(axis=1), 3600.0)
        self.assertGreaterEqual(durations.min(), 300.0)
        with self.assertRaisesRegex(ValueError, "too short"):
            _random_block_durations(1700.0, 6, 300.0, 1, np.random.default_rng(0))

    def test_labels_alternate_over_blocks_of_at_least_the_minimum(self):
        session, _ = synthetic_session()
        labels = random_block_contexts(session, 300, min_block_duration=300.0, seed=1)
        self.assertEqual(labels.shape, (300, 120))
        self.assertTrue(set(np.unique(labels)) <= {-1.0, 1.0})
        first_signs = set()
        for row in labels:
            switches = np.flatnonzero(np.diff(row) != 0) + 1
            self.assertLessEqual(switches.size, 5)
            # Trials are 30 s apart, so a block of >= 300 s holds >= 9 trial starts.
            block_trials = np.diff(np.r_[0, switches, row.size])
            self.assertGreaterEqual(block_trials.min(), 9)
            first_signs.add(row[0])
        self.assertEqual(first_signs, {-1.0, 1.0})
        np.testing.assert_array_equal(
            labels,
            random_block_contexts(session, 300, min_block_duration=300.0, seed=1),
        )
        self.assertFalse(
            np.array_equal(
                labels,
                random_block_contexts(session, 300, min_block_duration=300.0, seed=2),
            )
        )

    def test_block_count_defaults_to_observed_blocks(self):
        session, _ = synthetic_session(n_blocks=4)
        labels = random_block_contexts(session, 200, min_block_duration=60.0, seed=0)
        n_switches = (np.diff(labels, axis=1) != 0).sum(axis=1)
        self.assertEqual(n_switches.max(), 3)
        explicit = random_block_contexts(
            session, 200, min_block_duration=60.0, n_blocks=6, seed=0
        )
        self.assertEqual((np.diff(explicit, axis=1) != 0).sum(axis=1).max(), 5)

    def test_random_block_null_wraps_labels(self):
        session, _ = synthetic_session()
        null = random_block_null(session, 10, seed=4)
        self.assertEqual(null.method, "labels")
        np.testing.assert_array_equal(
            null.labels, random_block_contexts(session, 10, seed=4)
        )
        with self.assertRaisesRegex(ValueError, "too short"):
            random_block_contexts(session, 10, min_block_duration=700.0)


if __name__ == "__main__":
    unittest.main()

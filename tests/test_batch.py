import argparse
import pickle
import unittest
from unittest import mock

import numpy as np

from gain_glm import ModelData, ModelSpec, Signal, compile_design
from gain_glm.batch import compare_models, main, parse_dropout, parse_positive_float


class BatchTests(unittest.TestCase):
    def test_cli_dropout_syntax(self):
        self.assertEqual(parse_dropout("gain:context").remove_gains, ("context",))
        self.assertEqual(parse_dropout("group:face").groups, ("face",))
        self.assertEqual(
            parse_dropout("predictors:licks,rewards").remove_predictors,
            ("licks", "rewards"),
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_dropout("context")

    def test_positive_float_parser_rejects_invalid_dt(self):
        self.assertEqual(parse_positive_float("0.05"), 0.05)
        for value in ("0", "-0.1", "nan", "inf"):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                parse_positive_float(value)

    def test_cli_forwards_cv_options_and_overrides_model_dt(self):
        with mock.patch("gain_glm.batch.fit_session") as fit_session:
            main(
                [
                    "--nwb-path",
                    "session.nwb",
                    "--session-id",
                    "session",
                    "--output-dir",
                    "results",
                    "--folds",
                    "7",
                    "--fold-seed",
                    "13",
                    "--dt",
                    "0.05",
                ]
            )

        options = fit_session.call_args.kwargs
        self.assertEqual(options["cv"].folds, 7)
        self.assertEqual(options["cv"].seed, 13)
        self.assertEqual(options["model"].dt, 0.05)

    def test_prepared_design_is_process_serializable(self):
        trial_index = np.repeat(np.arange(3), 4)
        data = ModelData(
            dt=0.1,
            trial_index=trial_index,
            signals={"x": np.arange(trial_index.size)},
        )
        prepared = compile_design(
            ModelSpec(
                (Signal("x", window=(0, 0), n_basis=1),),
                name="synthetic",
                dt=0.1,
            ),
            data,
        )
        restored = pickle.loads(pickle.dumps(prepared))
        self.assertEqual(restored.spec.name, "synthetic")
        self.assertEqual(restored.base_blocks["x"].shape, (12, 1))

    def test_model_comparison_requires_a_shared_time_grid_and_fit_rows(self):
        predictor = (Signal("x", window=(0, 0), n_basis=1),)
        first = ModelSpec(predictor, name="first", dt=0.1)
        different_dt = ModelSpec(predictor, name="different_dt", dt=0.2)
        different_rows = ModelSpec(
            predictor,
            name="different_rows",
            dt=0.1,
            fit_window=(-0.1, 0.2),
            fit_events=("cue",),
        )

        with self.assertRaisesRegex(ValueError, "same dt"):
            compare_models("unused.nwb", (first, different_dt))
        with self.assertRaisesRegex(ValueError, "same fit window and events"):
            compare_models("unused.nwb", (first, different_rows))


if __name__ == "__main__":
    unittest.main()

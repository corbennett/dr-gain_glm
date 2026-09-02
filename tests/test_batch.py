import argparse
import pickle
import unittest

import numpy as np

from gain_glm import ModelData, ModelSpec, Signal, compile_design
from gain_glm.batch import compare_models, parse_dropout


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

import argparse
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from gain_glm import ModelData, ModelSpec, Signal, compile_design
from gain_glm.batch import (
    compare_models,
    fit_session,
    main,
    parse_dropout,
    parse_positive_float,
    parse_qc_filter,
)


class BatchTests(unittest.TestCase):
    def test_qc_filter_syntax_validation(self):
        predicate = "amplitude_cutoff <= 0.1 AND presence_ratio >= 0.7"
        self.assertEqual(parse_qc_filter(predicate), predicate)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_qc_filter("amplitude_cutoff <=")

    def test_cli_forwards_qc_selection(self):
        base = [
            "--nwb-path",
            "session.nwb",
            "--session-id",
            "session",
            "--output-dir",
            "results",
        ]
        for option, value in (
            ("--qc-column", "custom_qc"),
            ("--qc-filter", "amplitude_cutoff <= 0.1 AND presence_ratio >= 0.7"),
        ):
            with (
                self.subTest(option=option),
                mock.patch("gain_glm.batch.fit_session") as fit,
            ):
                main([*base, option, value])
            self.assertEqual(fit.call_args.kwargs[option[2:].replace("-", "_")], value)

    def test_fit_session_applies_qc_before_limit(self):
        predicate = "amplitude_cutoff <= 0.1"
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("gain_glm.batch.qc_unit_ids", return_value=["a", "b"]) as qc,
            mock.patch("gain_glm.batch.load_session") as load,
            mock.patch("gain_glm.batch.prepare"),
            mock.patch("gain_glm.batch.load_unit_target") as target,
            mock.patch("gain_glm.batch._evaluate_unit", return_value=("a", {})),
        ):
            path = fit_session(
                "session.nwb",
                "session",
                directory,
                qc_filter=predicate,
                limit=1,
                n_jobs=1,
            )
            payload = json.loads(path.read_text())
        qc.assert_called_once_with(
            "session.nwb", qc_column="default_qc", qc_filter=predicate
        )
        target.assert_called_once_with(load.return_value, "a")
        self.assertEqual(list(payload["units"]), ["a"])

    def test_comparison_qc_and_explicit_unit_override(self):
        model = ModelSpec((Signal("x", window=(0, 0), n_basis=1),), name="test", dt=0.1)
        predicate = "presence_ratio >= 0.7"
        for unit_ids in (None, ["explicit"]):
            with (
                self.subTest(unit_ids=unit_ids),
                mock.patch("gain_glm.batch.qc_unit_ids", return_value=["a", "b"]) as qc,
                mock.patch("gain_glm.batch.load_session"),
                mock.patch("gain_glm.batch.prepare") as prepare,
                mock.patch("gain_glm.batch.load_unit_target") as target,
            ):
                result = prepare.return_value.evaluate.return_value
                result.cv.r2_per_fold = np.array([0.1])
                result.cv.converged_per_fold = np.array([True])
                result.dropouts = {}
                with mock.patch("gain_glm.batch.pl.DataFrame"):
                    compare_models(
                        "session.nwb",
                        [model],
                        unit_ids=unit_ids,
                        unit_limit=1,
                        qc_filter=predicate,
                    )
                self.assertEqual(
                    target.call_args.args[1], "a" if unit_ids is None else "explicit"
                )
                self.assertEqual(target.call_count, 1)
                if unit_ids is None:
                    qc.assert_called_once_with(
                        "session.nwb", qc_column="default_qc", qc_filter=predicate
                    )
                else:
                    qc.assert_not_called()

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
                    "--cv-split",
                    "trials",
                    "--fold-seed",
                    "13",
                    "--dt",
                    "0.05",
                    "--use-instruction-trials",
                ]
            )

        options = fit_session.call_args.kwargs
        self.assertEqual(options["cv"].folds, 7)
        self.assertEqual(options["cv"].seed, 13)
        self.assertEqual(options["cv"].split, "trials")
        self.assertEqual(options["model"].dt, 0.05)
        self.assertTrue(options["use_instruction_trials"])

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

    def test_fit_session_logs_elapsed_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with (
                mock.patch("gain_glm.batch.qc_unit_ids", return_value=[]),
                mock.patch("gain_glm.batch.load_session"),
                mock.patch("gain_glm.batch.prepare"),
                mock.patch(
                    "gain_glm.batch.time.perf_counter",
                    side_effect=(10.0, 12.5),
                ),
            ):
                fit_session("session.nwb", "session", output_dir, n_jobs=1)

            result = json.loads((output_dir / "session.json").read_text())
            has_separate_runtime_file = (output_dir / "session_runtime.json").exists()

        self.assertEqual(result["session_id"], "session")
        self.assertEqual(result["runtime_seconds"], 2.5)
        self.assertFalse(has_separate_runtime_file)

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

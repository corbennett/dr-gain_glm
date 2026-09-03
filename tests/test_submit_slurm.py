import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.submit_slurm import (
    fit_command,
    pending_sessions,
    resolve_output_dir,
    save_run_params,
)


class SubmitSlurmTests(unittest.TestCase):
    def test_resolve_output_dir_accepts_explicit_path(self):
        self.assertEqual(
            resolve_output_dir("results/experiment-a"), Path("results/experiment-a")
        )

    def test_pending_sessions_excludes_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "done.json").touch()

            self.assertEqual(
                pending_sessions(["done", "new"], output_dir),
                ["new"],
            )
            self.assertEqual(
                pending_sessions(["done", "new"], output_dir, overwrite=True),
                ["done", "new"],
            )

    def test_fit_command_forwards_cv_and_dt_options(self):
        command = fit_command(
            "session.nwb",
            "session",
            Path("results"),
            model="default",
            max_iter=50,
            folds=7,
            fold_seed=0,
            dt=0.05,
        )

        self.assertIn("--folds 7", command)
        self.assertIn("--fold-seed 0", command)
        self.assertIn("--dt 0.05", command)

    def test_fit_command_omits_optional_cv_seed_and_dt(self):
        command = fit_command(
            "session.nwb",
            "session",
            Path("results"),
            model="default",
            max_iter=50,
            folds=5,
        )

        self.assertIn("--folds 5", command)
        self.assertNotIn("--fold-seed", command)
        self.assertNotIn("--dt", command)

    def test_run_params_record_overridden_dt_and_cv_options(self):
        args = argparse.Namespace(
            model="default",
            dt=0.05,
            folds=7,
            fold_seed=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_run_params(output_dir, args, ["session"])
            params = json.loads((output_dir / "run_params.json").read_text())

        self.assertEqual(params["model"]["dt"], 0.05)
        self.assertEqual(params["arguments"]["folds"], 7)
        self.assertEqual(params["arguments"]["fold_seed"], 0)


if __name__ == "__main__":
    unittest.main()

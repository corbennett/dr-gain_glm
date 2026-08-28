import tempfile
import unittest
from pathlib import Path

from scripts.submit_slurm import pending_sessions, resolve_output_dir


class SubmitSlurmTests(unittest.TestCase):
    def test_resolve_output_dir_accepts_explicit_path(self):
        self.assertEqual(resolve_output_dir("results/experiment-a"), Path("results/experiment-a"))

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


if __name__ == "__main__":
    unittest.main()

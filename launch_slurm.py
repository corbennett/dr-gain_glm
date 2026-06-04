"""Launch one SLURM job per session to fit the bilinear gain GLM.

Builds the session table (good-behavior DynamicRouting ephys sessions), then
submits a SLURM job per session. Each job runs run_fit.py on that session's NWB,
which fits every QC-pass unit and writes <OUTPUT_DIR>/<session_id>.json.

Usage:
    python launch_slurm.py              # submit a job per session
    python launch_slurm.py --dry-run    # print the sbatch scripts, submit nothing
    python launch_slurm.py --limit 3    # submit only the first 3 sessions (testing)

Edit the CONFIG block below for your cluster (venv python, resources, output dir).
"""

import argparse
import pathlib

import polars as pl
import upath
from simple_slurm import Slurm

# ----------------------------------------------------------------------------
# CONFIG — edit for your cluster
# ----------------------------------------------------------------------------
# Repo root (this file's directory) — run_fit.py lives here.
REPO_DIR = pathlib.Path(__file__).resolve().parent
RUN_FIT = REPO_DIR / "run_fit.py"

# Absolute path to the venv's python interpreter on the compute nodes. simple_slurm
# uses this to run run_fit.py (no module load / conda activate needed). Defaults to
# this checkout's .venv; override if the compute nodes use a different path.
VENV_PYTHON = str(REPO_DIR / ".venv/bin/python")

# Where per-session result JSONs are written (shared path visible to all nodes).
OUTPUT_DIR = "/path/to/results"

# Directory for SLURM stdout/stderr logs.
LOG_DIR = "logs"

# SLURM resource requests per job.
PARTITION = None          # e.g. "braintv"; None lets the cluster default decide
CPUS_PER_TASK = 8
MEM = "32G"
TIME = "4:00:00"          # H:MM:SS walltime


def get_session_table() -> pl.DataFrame:
    """Good-behavior DynamicRouting ephys sessions joined to their NWB paths.

    Mirrors the first cell of scratch.ipynb.
    """
    nwb_dir = upath.UPath(
        "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.272/", anon=True
    )
    nwb_paths_df = pl.DataFrame([
        {'nwb_path': p.as_posix(), 'session_id': p.stem}
        for p in nwb_dir.iterdir()
        if p.suffix == '.nwb'
    ])
    return (
        pl.read_parquet(
            "s3://aind-scratch-data/dynamic-routing/session_metadata/session_table.parquet",
            storage_options={'skip_signature': 'true'},
        )
        .filter(
            'is_good_behavior',
            pl.col('project') == 'DynamicRouting',
            'is_ephys',
            ~pl.col('is_naive'),
            ~pl.col('is_context_naive'),
            'is_task',
            ~pl.col('is_opto_perturbation'),
            ~pl.col('is_injection_perturbation'),
            ~pl.col('is_opto_control'),
            ~pl.col('is_injection_control'),
        )
        .join(nwb_paths_df, on='session_id', how='inner')
        .select('session_id', 'nwb_path')
    )


def make_slurm(session_id: str) -> Slurm:
    """A configured Slurm job for one session."""
    kwargs = dict(
        job_name=f"glm_{session_id}",
        cpus_per_task=CPUS_PER_TASK,
        mem=MEM,
        time=TIME,
        output=f"{LOG_DIR}/%x_%j.out",
        error=f"{LOG_DIR}/%x_%j.err",
    )
    if PARTITION:
        kwargs["partition"] = PARTITION
    return Slurm(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the sbatch scripts but submit nothing.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Submit only the first N sessions (for testing).")
    parser.add_argument("--unit-limit", type=int, default=None,
                        help="Pass --limit N to run_fit.py (fit only N units/session).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Pass --overwrite to run_fit.py (refit existing outputs).")
    args = parser.parse_args()

    pathlib.Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    sessions = get_session_table()
    if args.limit is not None:
        sessions = sessions.head(args.limit)
    print(f"{len(sessions)} sessions to submit")

    for row in sessions.iter_rows(named=True):
        session_id, nwb_path = row["session_id"], row["nwb_path"]
        cmd = (
            f"{VENV_PYTHON} {RUN_FIT} "
            f"--nwb-path {nwb_path} "
            f"--session-id {session_id} "
            f"--output-dir {OUTPUT_DIR}"
        )
        if args.unit_limit is not None:
            cmd += f" --limit {args.unit_limit}"
        if args.overwrite:
            cmd += " --overwrite"

        slurm = make_slurm(session_id)
        if args.dry_run:
            print(f"\n# ===== {session_id} =====")
            print(slurm)
            print(cmd)
        else:
            job_id = slurm.sbatch(cmd)
            print(f"submitted {session_id}: job {job_id}")


if __name__ == "__main__":
    main()

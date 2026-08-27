"""Submit one ``gain-glm-fit`` SLURM job per Dynamic Routing session.

Cluster paths and resources are intentionally collected in the configuration
block below. Use ``--dry-run`` to inspect commands before submission.
"""

import argparse
from pathlib import Path

import polars as pl
import upath
from simple_slurm import Slurm

REPO_DIR = Path(__file__).resolve().parents[1]
PYTHON = str(REPO_DIR / ".venv/bin/python")
OUTPUT_DIR = "/path/to/results"
LOG_DIR = REPO_DIR / "logs"
PARTITION = None
CPUS = 8
MEMORY = "32G"
WALLTIME = "4:00:00"


def sessions() -> pl.DataFrame:
    nwb_dir = upath.UPath(
        "s3://aind-scratch-data/dynamic-routing/cache/nwb/v0.0.272/", anon=True
    )
    paths = pl.DataFrame(
        {
            "nwb_path": path.as_posix(),
            "session_id": path.stem,
        }
        for path in nwb_dir.iterdir()
        if path.suffix == ".nwb"
    )
    return (
        pl.read_parquet(
            "s3://aind-scratch-data/dynamic-routing/session_metadata/session_table.parquet",
            storage_options={"skip_signature": "true"},
        )
        .filter(
            "is_good_behavior",
            pl.col("project") == "DynamicRouting",
            "is_ephys",
            ~pl.col("is_naive"),
            ~pl.col("is_context_naive"),
            "is_task",
            ~pl.col("is_opto_perturbation"),
            ~pl.col("is_injection_perturbation"),
            ~pl.col("is_opto_control"),
            ~pl.col("is_injection_control"),
        )
        .join(paths, on="session_id", how="inner")
        .select("session_id", "nwb_path")
    )


def slurm_job(session_id: str) -> Slurm:
    options = {
        "job_name": f"glm_{session_id}",
        "cpus_per_task": CPUS,
        "mem": MEMORY,
        "time": WALLTIME,
        "output": str(LOG_DIR / "%x_%j.out"),
        "error": str(LOG_DIR / "%x_%j.err"),
    }
    if PARTITION:
        options["partition"] = PARTITION
    return Slurm(**options)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--unit-limit", type=int)
    parser.add_argument("--model", default="default")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    table = sessions().head(args.limit) if args.limit else sessions()
    for row in table.iter_rows(named=True):
        command = (
            f"{PYTHON} -m gain_glm.batch "
            f"--nwb-path {row['nwb_path']} --session-id {row['session_id']} "
            f"--output-dir {OUTPUT_DIR} --model {args.model}"
        )
        if args.unit_limit:
            command += f" --limit {args.unit_limit}"
        if args.overwrite:
            command += " --overwrite"
        job = slurm_job(row["session_id"])
        if args.dry_run:
            print(job)
            print(command)
        else:
            print(f"submitted {row['session_id']}: {job.sbatch(command)}")


if __name__ == "__main__":
    main()

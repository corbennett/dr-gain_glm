"""Submit one ``gain-glm-fit`` SLURM job per Dynamic Routing session.

Cluster paths and resources are intentionally collected in the configuration
block below. Use ``--dry-run`` to inspect commands before submission.
"""

import argparse
import json
from dataclasses import asdict
from dataclasses import replace as dataclass_replace
from datetime import datetime
from pathlib import Path

import polars as pl
from dr_datacube import datacube_config, get_session_ids_from_github, list_nwb_sources
from simple_slurm import Slurm

from gain_glm import FitConfig
from gain_glm.batch import parse_positive_float
from gain_glm.dynamic_routing import MODELS

REPO_DIR = Path(__file__).resolve().parents[1]
PYTHON = str(REPO_DIR / ".venv/bin/python")
OUTPUT_DIR = "/allen//programs/mindscope/workgroups/dynamicrouting/corbettb/gain_glm"
LOG_DIR = REPO_DIR / "logs"
PARTITION = "braintv"
CPUS = 8
MEMORY = "32G"
WALLTIME = "4:00:00"

datacube_config.use_cache = True


def sessions() -> pl.DataFrame:
    good_session_ids = set(get_session_ids_from_github("brainwide"))
    paths = pl.DataFrame(
        {
            "nwb_path": nwb_path,
            "session_id": nwb_path.rsplit("/", 1)[-1].split(".", 1)[0],
        }
        for nwb_path in list_nwb_sources()
        if nwb_path.rsplit("/", 1)[-1].split(".", 1)[0] in good_session_ids
    )
    return paths.select("session_id", "nwb_path")


def pending_sessions(
    session_ids: list[str], output_dir: Path, *, overwrite: bool = False
) -> list[str]:
    """Return sessions whose output file does not already exist."""
    if overwrite:
        return session_ids
    return [
        session_id
        for session_id in session_ids
        if not (output_dir / f"{session_id}.json").exists()
    ]


def resolve_output_dir(output_dir: str | None) -> Path:
    """Use an explicit output directory or the date-based default."""
    if output_dir is not None:
        return Path(output_dir)
    return Path(OUTPUT_DIR) / datetime.now().astimezone().strftime("%m%d%y")


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


def fit_command(
    nwb_path: str,
    session_id: str,
    output_dir: Path,
    *,
    model: str,
    max_iter: int,
    folds: int,
    fold_seed: int | None = None,
    dt: float | None = None,
    unit_limit: int | None = None,
    overwrite: bool = False,
) -> str:
    """Build the batch-fitting command run inside one SLURM job."""
    command = (
        "export LAZYNWB_CATALOG_CACHE_PATH="
        '"${SLURM_TMPDIR:-/tmp}/lazynwb/catalog-${SLURM_JOB_ID}.sqlite"; '
        f"{PYTHON} -m gain_glm.batch "
        f"--nwb-path {nwb_path} --session-id {session_id} "
        f"--output-dir {output_dir} --model {model} "
        f"--max-iter {max_iter} --folds {folds}"
    )
    if fold_seed is not None:
        command += f" --fold-seed {fold_seed}"
    if dt is not None:
        command += f" --dt {dt}"
    if unit_limit is not None:
        command += f" --limit {unit_limit}"
    if overwrite:
        command += " --overwrite"
    return command


def save_run_params(
    output_dir: Path,
    args: argparse.Namespace,
    session_ids: list[str],
) -> None:
    model = MODELS[args.model]
    if args.dt is not None:
        model = dataclass_replace(model, dt=args.dt)
    params = {
        "output_dir": str(output_dir),
        "arguments": vars(args),
        "session_ids": session_ids,
        "model": asdict(model),
        "dropouts": [asdict(dropout) for dropout in model.dropouts],
    }
    with (output_dir / "run_params.json").open("w") as stream:
        json.dump(params, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--unit-limit", type=int)
    parser.add_argument("--model", choices=MODELS, default="default")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int)
    parser.add_argument(
        "--dt",
        type=parse_positive_float,
        help="Override the selected model's time-bin width in seconds",
    )
    parser.add_argument("--max-iter", type=int, default=FitConfig().max_iter)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output-dir",
        help=f"Output directory (default: {OUTPUT_DIR}/<MMDDYY>)",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_output_dir = resolve_output_dir(args.output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    table = sessions().head(args.limit) if args.limit else sessions()
    session_ids = pending_sessions(
        table["session_id"].to_list(), run_output_dir, overwrite=args.overwrite
    )
    if len(session_ids) != table.height:
        print(
            f"skipping {table.height - len(session_ids)} sessions with existing output"
        )
        table = table.filter(pl.col("session_id").is_in(session_ids))
    for row in table.iter_rows(named=True):
        command = fit_command(
            row["nwb_path"],
            row["session_id"],
            run_output_dir,
            model=args.model,
            max_iter=args.max_iter,
            folds=args.folds,
            fold_seed=args.fold_seed,
            dt=args.dt,
            unit_limit=args.unit_limit,
            overwrite=args.overwrite,
        )
        job = slurm_job(row["session_id"])
        if args.dry_run:
            print(job)
            print(command)
        else:
            print(f"submitted {row['session_id']}: {job.sbatch(command)}")
    if not args.dry_run:
        save_run_params(run_output_dir, args, table["session_id"].to_list())


if __name__ == "__main__":
    main()

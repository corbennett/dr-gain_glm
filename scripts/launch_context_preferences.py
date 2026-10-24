"""Launch the full analysis under macOS caffeinate, with a persistent log."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/modality_context/full_expanded"),
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=Path("results/modality_context/discovery")
    )
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--anatomy", type=Path)
    parser.add_argument("--include-areas", nargs="*", default=None)
    args = parser.parse_args()
    caffeinate = shutil.which("caffeinate")
    if caffeinate is None:
        raise RuntimeError("This launcher requires macOS caffeinate")
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    launch_path = output / "launch.json"
    if launch_path.exists():
        previous = json.loads(launch_path.read_text())
        check = subprocess.run(
            ["ps", "-p", str(previous["pid"]), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
        if "gain_glm.context_preferences" in check.stdout:
            raise RuntimeError(f"An analysis is already running: PID {previous['pid']}")
    command = [
        caffeinate,
        "-i",
        sys.executable,
        "-u",
        "-m",
        "gain_glm.context_preferences",
        "--output-dir",
        str(output),
        "--metadata-dir",
        str(args.metadata_dir.resolve()),
        "--permutations",
        str(args.permutations),
        "--threads",
        str(args.threads),
    ]
    if args.anatomy:
        command.extend(["--anatomy", str(args.anatomy.resolve())])
    if args.include_areas is not None:
        command.extend(["--include-areas", *args.include_areas])
    log_path = output / "run.log"
    with log_path.open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=os.environ.copy(),
        )
    record = {
        "pid": process.pid,
        "started_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "log": str(log_path),
    }
    launch_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

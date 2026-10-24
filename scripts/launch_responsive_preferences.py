"""Launch the responsive-cell analysis under macOS caffeinate."""

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
        "--source-dir",
        type=Path,
        default=Path("results/modality_context/full_expanded"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/responsive_preferences/full")
    )
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    caffeinate = shutil.which("caffeinate")
    if not caffeinate:
        raise RuntimeError("This launcher requires macOS caffeinate")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    launch = out / "launch.json"
    if launch.exists():
        previous = json.loads(launch.read_text())
        check = subprocess.run(
            ["ps", "-p", str(previous["pid"]), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
        if "gain_glm.responsive_preferences" in check.stdout:
            raise RuntimeError(f"Already running: PID {previous['pid']}")
    command = [
        caffeinate,
        "-i",
        sys.executable,
        "-u",
        "-m",
        "gain_glm.responsive_preferences",
        "--source-dir",
        str(args.source_dir.resolve()),
        "--output-dir",
        str(out),
        "--permutations",
        str(args.permutations),
        "--threads",
        str(args.threads),
    ]
    with (out / "run.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
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
        "log": str(out / "run.log"),
    }
    launch.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

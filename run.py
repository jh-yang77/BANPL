import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
ENTRIES = {
    "generate": ("scripts/generate_banpl_neg_images.py", []),
    "train": ("train_banpl_seed1.py", ["--train", "1"]),
    "eval": ("train_banpl_seed1.py", ["--eval-only"]),
}


def main():
    parser = argparse.ArgumentParser(description="Run the BANPL main experiment.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("task", choices=ENTRIES)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    script, defaults = ENTRIES[args.task]
    command = [args.python, str(ROOT / script), *defaults, *forwarded]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "Dassl.pytorch"), environment.get("PYTHONPATH", "")]
    )
    environment.setdefault("BANPL_DATA_ROOT", str(ROOT / "data"))
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.run(command, cwd=ROOT, env=environment).returncode)


if __name__ == "__main__":
    main()

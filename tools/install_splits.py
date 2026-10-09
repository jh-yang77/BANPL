import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import pickle
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "resources"


class SplitRecord:
    pass


class SplitUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("dassl.data.datasets.base_dataset", "Datum"):
            return SplitRecord
        raise pickle.UnpicklingError(f"Unexpected pickle class: {module}.{name}")


def relative_path(value):
    if not isinstance(value, str):
        raise ValueError("Image paths must be strings")
    relative = value.split("/imagenet/", 1)[-1]
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 3 or path.parts[0] != "train":
        raise ValueError(f"Invalid training-image path: {value}")
    return path.as_posix()


def read_records(path, require_relative=False):
    with Path(path).open("rb") as stream:
        data = SplitUnpickler(stream).load()
    if not isinstance(data, dict) or set(data) != {"train"}:
        raise ValueError("Expected a split with one 'train' list")
    records = []
    for item in data["train"]:
        if not isinstance(item, SplitRecord) or set(vars(item)) != {"_impath", "_label", "_domain", "_classname"}:
            raise ValueError("Unexpected split record")
        image_path = relative_path(item._impath)
        if require_relative and item._impath != image_path:
            raise ValueError("Public split files must use paths relative to imagenet/")
        records.append((image_path, int(item._label), item._classname, int(item._domain)))
    return records


def verified_records(seed):
    metadata = json.loads((RESOURCES / "splits_provenance.json").read_text())
    record = next(item for item in metadata if item["seed"] == seed)
    path = RESOURCES / "splits" / f"shot_16-seed_{seed}.pkl"
    csv_path = RESOURCES / f"imagenet1k_16shot_seed{seed}.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["release_pickle_sha256"]:
        raise ValueError(f"Split checksum mismatch: {path}")
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != record["csv_sha256"]:
        raise ValueError(f"CSV checksum mismatch: {csv_path}")
    records = read_records(path, require_relative=True)
    with csv_path.open(newline="") as stream:
        readable = [(relative_path(row["path"]), int(row["label"]), row["classname"], 0) for row in csv.DictReader(stream)]
    if records != readable:
        raise ValueError(f"PKL and CSV sample order or labels differ for seed {seed}")
    counts = Counter(item[1] for item in records)
    if len(records) != 16000 or set(counts) != set(range(1000)) or set(counts.values()) != {16}:
        raise ValueError(f"Invalid 16-shot split for seed {seed}")
    if len({item[0] for item in records}) != len(records):
        raise ValueError(f"Duplicate training images in seed {seed}")
    return records


def install_split(data_root, output, seed, records):
    image_root = data_root / "imagenet"
    paths = [image_root / item[0] for item in records]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} selected images; first: {missing[0]}")
    target = output / f"shot_16-seed_{seed}.pkl"
    if target.exists():
        if read_records(target) != records:
            raise FileExistsError(f"{target} contains another split; choose a new --output-dir")
        with target.open("rb") as stream:
            old = SplitUnpickler(stream).load()["train"]
        if [item._impath for item in old] != [str(path) for path in paths]:
            raise FileExistsError(f"{target} has different machine paths; choose a new --output-dir")
        print(f"Already matches: {target}")
        return
    sys.path.insert(0, str(ROOT / "Dassl.pytorch"))
    from dassl.data.datasets import Datum

    items = [Datum(impath=str(path), label=record[1], classname=record[2], domain=record[3]) for path, record in zip(paths, records)]
    output.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        pickle.dump({"train": items}, stream, protocol=pickle.HIGHEST_PROTOCOL)
    if read_records(target) != records:
        raise ValueError(f"Installed split verification failed: {target}")
    print(f"Installed {target}: {len(items)} images")


def main():
    parser = argparse.ArgumentParser(description="Install the BANPL 16-shot splits with local image paths.")
    parser.add_argument("--root", type=Path, help="Parent directory of imagenet/")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verified = {seed: verified_records(seed) for seed in args.seeds}
    for seed, records in verified.items():
        print(f"Verified seed {seed}: {len(records)} images, 1,000 classes, 16 per class")
    if args.verify_only:
        return
    if args.root is None:
        parser.error("--root is required unless --verify-only is used")
    root = args.root.expanduser().resolve()
    classnames = root / "imagenet/classnames.txt"
    expected = (RESOURCES / "imagenet_classnames.txt").read_text().strip()
    if classnames.exists() and classnames.read_text().strip() != expected:
        raise ValueError(f"Class-name mapping differs: {classnames}")
    if not classnames.exists():
        classnames.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RESOURCES / "imagenet_classnames.txt", classnames)
    output = args.output_dir.expanduser().resolve() if args.output_dir else root / "imagenet/split_fewshot"
    for seed, records in verified.items():
        install_split(root, output, seed, records)


if __name__ == "__main__":
    main()

"""Check that the data root contains every dataset GRACE uses, and say how to get the missing ones.

The data root is `--root`, else the GRACE_DATA environment variable, else `grace_data/` in the
current directory (as in `configs/defaults.yaml`). Each dataset is checked for its expected files
and number of images/videos; `--quick` only checks that they exist. Exits with status 1 if
anything is missing or incomplete.

Usage:
    python -m scripts.check_data
    python -m scripts.check_data --only coco imagenet-val
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

DOWNLOAD = "python -m scripts.download_data --only {}"


@dataclass(frozen=True)
class File:
    path: str
    lines: int | None = None  # expected number of lines (text files)


@dataclass(frozen=True)
class Dir:
    path: str
    files: int  # expected number of files with one of `suffixes`, recursively
    suffixes: tuple[str, ...]


@dataclass(frozen=True)
class Dataset:
    name: str
    used_for: str
    how: str  # how to get it
    checks: list[File | Dir] = field(default_factory=list)


EGO4D_FEATURES = "Not distributed. Extract PE-Core-L14-336 features from the Ego4D videos (license at https://ego4ddataset.com) with `python -m scripts.extract_ego4d_features` (see DATA.md)."

DATASETS = [
    Dataset(
        "ego4d-annotations",
        "EgoMCQ eval (egomcq.json), video training (egoclip.csv)",
        DOWNLOAD.format("ego4d"),
        [File("ego4d/annotations/egomcq.json"), File("ego4d/annotations/egoclip.csv")],
    ),
    Dataset(
        "ego4d-metadata",
        "video training (scenario tags)",
        "Requires the Ego4D license (https://ego4ddataset.com). The Ego4D CLI "
        "(https://ego4d-data.org/docs/start-here/) downloads the `ego4d.json` metadata file along with "
        "any dataset; copy it to ego4d/annotations/ego4d.json.",
        [File("ego4d/annotations/ego4d.json")],
    ),
    Dataset("egomcq-features", "EgoMCQ eval", EGO4D_FEATURES, [File("ego4d/features/EgoMCQ_PE-Core-L14-336/features.pt")]),
    Dataset("egoclip-features", "video training", EGO4D_FEATURES, [File("ego4d/features/EgoClip_PE-Core-L14-336/features.pt")]),
    Dataset(
        "charades_ego",
        "Charades-Ego eval",
        DOWNLOAD.format("charades_ego"),
        [
            File("charades_ego/annotations/Charades_v1_classes.txt", lines=157),
            File("charades_ego/annotations/CharadesEgo_v1_test_only1st.csv"),
            Dir("charades_ego/videos", 7860, (".mp4",)),
        ],
    ),
    Dataset(
        "convisbench",
        "ConVisBench eval",
        DOWNLOAD.format("convisbench"),
        [File("convisbench/ConVIS.jsonl"), Dir("convisbench/videos", 543, (".mp4",))],
    ),
    Dataset(
        "coco",
        "COCO retrieval eval",
        DOWNLOAD.format("coco"),
        [File("coco/annotations/captions_val2017.json"), Dir("coco/val2017", 5000, (".jpg",))],
    ),
    Dataset(
        "flickr30k",
        "Flickr30k retrieval eval",
        "Request the dataset at https://shannon.cs.illinois.edu/DenotationGraph/. Put the 31,783 images "
        "in flickr30k/Images/ and the captions in flickr30k/flickr30k_all.tsv, one `<image>#<i>\\t<caption>` "
        "line per caption (the format of the official results_20130124.token).",
        [File("flickr30k/flickr30k_all.tsv", lines=158915), Dir("flickr30k/Images", 31783, (".jpg",))],
    ),
    Dataset(
        "imagenet-val",
        "ImageNet classification eval (incl. class folds)",
        "Download ILSVRC2012 from https://image-net.org/download.php (account and terms required). Put "
        "the validation images in imagenet/val/<wnid>/, one folder per class (torchvision ImageFolder layout).",
        [Dir("imagenet/val", 50000, (".jpeg",))],
    ),
    Dataset(
        "imagenet-train",
        "image training (labels of the precomputed features)",
        "As imagenet-val, with the training images in imagenet/train/<wnid>/.",
        [Dir("imagenet/train", 1281167, (".jpeg",))],
    ),
    Dataset(
        "imagenet-features",
        "image training",
        "Not distributed. Extract the CLIP ViT-B/32 features of ImageNet train (needs imagenet-train) with `python -m scripts.extract_imagenet_features`.",
        [File("imagenet/train_features_clip_ViT-B-32.pt")],
    ),
    Dataset(
        "flowers102",
        "Flowers102 classification eval",
        DOWNLOAD.format("flowers102"),
        [File("flowers-102/imagelabels.mat"), File("flowers-102/setid.mat"), Dir("flowers-102/jpg", 8189, (".jpg",))],
    ),
    Dataset(
        "food101",
        "Food101 classification eval",
        DOWNLOAD.format("food101"),
        [File("food-101/meta/test.json"), Dir("food-101/images", 101000, (".jpg",))],
    ),
    Dataset(
        "ade20k",
        "ADE20k concept localization (TIPSv2)",
        DOWNLOAD.format("ade20k"),
        [
            File("ade20k/ADEChallengeData2016/objectInfo150.txt"),
            Dir("ade20k/ADEChallengeData2016/images/training", 20210, (".jpg",)),
            Dir("ade20k/ADEChallengeData2016/annotations/training", 20210, (".png",)),
            Dir("ade20k/ADEChallengeData2016/images/validation", 2000, (".jpg",)),
            Dir("ade20k/ADEChallengeData2016/annotations/validation", 2000, (".png",)),
        ],
    ),
]


def count_files(root: Path, suffixes: tuple[str, ...]) -> int:
    return sum(1 for _, _, files in os.walk(root) for f in files if f.lower().endswith(suffixes))


def problems(root: Path, check: File | Dir, quick: bool) -> list[str]:
    path = root / check.path
    if isinstance(check, File):
        if not path.is_file():
            return [f"{check.path}: missing"]
        if not quick and check.lines is not None:
            with open(path, "rb") as f:
                lines = sum(1 for line in f if line.strip())
            if lines != check.lines:
                return [f"{check.path}: {lines} lines, expected {check.lines}"]
        return []
    if not path.is_dir():
        return [f"{check.path}/: missing"]
    if not quick and (found := count_files(path, check.suffixes)) != check.files:
        return [f"{check.path}/: {found} {'/'.join(check.suffixes)} files, expected {check.files}"]
    return []


def main() -> None:
    names = [d.name for d in DATASETS]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("GRACE_DATA", "grace_data")), help="data root (default: $GRACE_DATA, else grace_data/)")
    parser.add_argument("--only", nargs="+", choices=names, default=names, help="datasets to check (default: all)")
    parser.add_argument("--quick", action="store_true", help="only check that files and directories exist, don't count them")
    args = parser.parse_args()

    print(f"Data root: {args.root.resolve()}\n")
    failed = []
    for dataset in (d for d in DATASETS if d.name in args.only):
        issues = [issue for check in dataset.checks for issue in problems(args.root, check, args.quick)]
        print(f"  {'OK' if not issues else 'MISSING':8} {dataset.name:18} {dataset.used_for}")
        for issue in issues:
            print(f"  {'':8} {'':18} - {issue}")
        if issues:
            failed.append(dataset)

    if failed:
        print("\nHow to get the missing datasets:")
        for dataset in failed:
            print(f"  {dataset.name}: {dataset.how}")
        sys.exit(1)
    print("\nAll datasets are in place.")


if __name__ == "__main__":
    main()

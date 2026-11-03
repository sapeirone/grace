"""Download the publicly available datasets used by GRACE into the data root.

The data root is `--root`, else the GRACE_DATA environment variable, else `grace_data/` in the
current directory (the configs resolve it the same way, see `configs/defaults.yaml`). Each dataset
is skipped if it's already in place, so the script can be re-run after an interruption.

Data that needs a license or a login (ImageNet, Flickr30k, Ego4D metadata) and the precomputed
features can't be downloaded here: `python -m scripts.check_data` lists what is missing and how to
get it.

Usage:
    python -m scripts.download_data                      # everything
    python -m scripts.download_data --only coco ade20k   # a subset
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path


def gdrive_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


CHARADES_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades"
COCO_URL = "http://images.cocodataset.org"
CONVISBENCH_REPO = "bliberatori/ConViS-Bench"
CONVISBENCH_REVISION = "8fa5ed88078134095103ac43461922697e8a1079"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path, md5: str | None = None) -> Path:
    """Download `url` to `dest` with curl and check its md5. The download goes to `<dest>.part` first
    (resumed if a previous run was interrupted), so an existing `dest` is always complete."""
    if dest.exists() and (md5 is None or md5sum(dest) == md5):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    print(f"Downloading {url} -> {dest}")
    subprocess.run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(partial), url], check=True)
    if md5 is not None and (actual := md5sum(partial)) != md5:
        partial.unlink()
        raise RuntimeError(f"{dest.name}: md5 {actual} != expected {md5}; deleted, re-run to retry")
    partial.rename(dest)
    return dest


def extract(archive: Path, into: Path, members: list[str] | None = None) -> None:
    """Extract `archive` (zip or tar) into `into`, optionally only `members`."""
    print(f"Extracting {archive.name} -> {into}")
    into.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into, members=members)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(into, members=None if members is None else [tf.getmember(m) for m in members], filter="data")


class Downloader:
    def __init__(self, root: Path, keep_archives: bool):
        self.root = root
        self.archives = root / ".downloads"
        self.keep_archives = keep_archives

    def archive(self, url: str, md5: str | None = None) -> Path:
        return fetch(url, self.archives / url.rsplit("/", 1)[-1], md5)

    def cleanup(self, *paths: Path) -> None:
        if not self.keep_archives:
            for path in paths:
                path.unlink(missing_ok=True)

    def ego4d(self) -> None:
        """EgoClip / EgoMCQ annotations, as released by EgoVLP (https://github.com/showlab/EgoVLP)."""
        anns = self.root / "ego4d" / "annotations"
        fetch(gdrive_url("1-aaDu_Gi-Y2sQI_2rsI2D1zvQBJnHpXl"), anns / "egoclip.csv", md5="37573cf4ee59d11e16d0f095cd04113f")
        fetch(gdrive_url("1-5iRYf4BCHmj4MYQYFRMY4bhsWJUN3rW"), anns / "egomcq.json", md5="ef73900e62bb3938e9f0177ee39ae11f")

    def charades_ego(self) -> None:
        """Charades-Ego annotations and videos (480p release)."""
        target = self.root / "charades_ego"
        if not (target / "annotations" / "CharadesEgo_v1_test_only1st.csv").exists():
            archive = self.archive(f"{CHARADES_URL}/CharadesEgo.zip")
            extract(archive, target)
            (target / "CharadesEgo").rename(target / "annotations")
            self.cleanup(archive)
        if not (target / "videos").exists():
            archive = self.archive(f"{CHARADES_URL}/CharadesEgo_v1_480.tar")
            extract(archive, target)
            (target / "CharadesEgo_v1_480").rename(target / "videos")
            self.cleanup(archive)

    def convisbench(self) -> None:
        """ConViS-Bench video pairs and human annotations, from Hugging Face."""
        from huggingface_hub import snapshot_download

        snapshot_download(CONVISBENCH_REPO, repo_type="dataset", revision=CONVISBENCH_REVISION, local_dir=self.root / "convisbench")

    def coco(self) -> None:
        """COCO 2017 validation images and captions."""
        target = self.root / "coco"
        if not (target / "val2017").exists():
            archive = self.archive(f"{COCO_URL}/zips/val2017.zip", md5="442b8da7639aecaf257c1dceb8ba8c80")
            extract(archive, target)
            self.cleanup(archive)
        if not (target / "annotations" / "captions_val2017.json").exists():
            archive = self.archive(f"{COCO_URL}/annotations/annotations_trainval2017.zip", md5="f4bbac642086de4f52a3fdda2de5fa2c")
            extract(archive, target, members=["annotations/captions_val2017.json"])
            self.cleanup(archive)

    def ade20k(self) -> None:
        """ADE20k SceneParsing (ADEChallengeData2016)."""
        target = self.root / "ade20k"
        if not (target / "ADEChallengeData2016" / "objectInfo150.txt").exists():
            archive = self.archive("http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip")
            extract(archive, target)
            self.cleanup(archive)

    def flowers102(self) -> None:
        """Oxford Flowers102, via torchvision (layout `<root>/flowers-102`)."""
        from torchvision.datasets import Flowers102

        Flowers102(str(self.root), split="test", download=True)
        self.cleanup(self.root / "flowers-102" / "102flowers.tgz")

    def food101(self) -> None:
        """Food101, via torchvision (layout `<root>/food-101`)."""
        from torchvision.datasets import Food101

        Food101(str(self.root), split="test", download=True)
        self.cleanup(self.root / "food-101.tar.gz")


DATASETS = ["ego4d", "charades_ego", "convisbench", "coco", "ade20k", "flowers102", "food101"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("GRACE_DATA", "grace_data")), help="data root (default: $GRACE_DATA, else grace_data/)")
    parser.add_argument("--only", nargs="+", choices=DATASETS, default=DATASETS, help="datasets to download (default: all)")
    parser.add_argument("--keep-archives", action="store_true", help="keep the downloaded archives in <root>/.downloads")
    args = parser.parse_args()

    if shutil.which("curl") is None:
        raise SystemExit("curl is required")
    downloader = Downloader(args.root, args.keep_archives)
    for name in args.only:
        print(f"== {name}")
        getattr(downloader, name)()
    if not args.keep_archives and downloader.archives.exists() and not any(downloader.archives.iterdir()):
        downloader.archives.rmdir()
    print(f"Done. Check the data root with: python -m scripts.check_data --root {args.root}")


if __name__ == "__main__":
    main()

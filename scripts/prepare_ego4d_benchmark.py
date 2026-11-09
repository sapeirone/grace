from argparse import ArgumentParser
import sys
import pandas as pd
from pathlib import Path
import os
import subprocess

"""
This module selects a subset of the ego4d dataset.
To avoid downloading huge videos we set a cap on their duration.
To correctly understand the goodness of GRACE + eligibility traces on both temporal and space grounding there are two key arguments:
- minDur: this indicates the minimum duration of a clip. This is essential because providing a clip that is too short neglects the 
possibility to correctly evaluate the eligbility traces
- minObj: this indicates the minimum number of objects inside the clip. We want this number to be at least > 0.

In conclusion: this module prepares the data, by selecting the videos to be downloaded and to be passed to SAM3 to generate the benchmark.
"""


CHECK_DATA = "python -m scripts.check_data --only ego4d"

def parse_args(args = sys.argv[1:]):
    parser = ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("GRACE_DATA", "grace_data")), help="data root (default: $GRACE_DATA, else grace_data/)")
    parser.add_argument("--nSamples", type=int, default=100, help="number of samples (which is different from number of videos)")
    parser.add_argument("--minDur", type=float, default=5.0, help="minimum duration of the sample (unit of measure: seconds)")
    parser.add_argument("--minObj", type=int, default=1, help="minimum number of objects")
    parser.add_argument("--maxVidDur", type=float, default=600, help="maximum duration of the whole video (unit of measure: seconds)")
    return parser.parse_args(args)

def check_args(args):
    if args.minDur <= 3.0: raise ValueError("Please set a higher minimum duration (minimum = 5s)")
    if args.minObj <= 0: raise ValueError("Please set a higher minimum number of objects (minimum = 1)")
    if args.maxVidDur <= 3.0 or args.maxVidDur > 900: raise ValueError("Please set a proper maximum video duration ([3.0, 900.0])")

def clean_df(df: pd.DataFrame, args) -> pd.DataFrame:
    """
    It cleans the dataframe by executing the following steps:
        - it correctly casts the numeric columns
        - it drops the rows with null key values
        - it filters the dataframe based on the necessities (>= min duration and >= min objects)
        - it filters very long videos
    """
    for col in ["video_dur", "narration_time", "clip_start", "clip_end"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["video_dur", "clip_start", "clip_end", "tag_noun"])

    n_objects = df["tag_noun"].str.count(",") + 1
    n_objects = n_objects.where(df["tag_noun"] != "[]", 0)

    filtered_df = df[
        ((df["clip_end"] - df["clip_start"]) >= args.minDur)
        & (n_objects >= args.minObj)
        & (df["video_dur"] <= args.maxVidDur)
    ].copy()

    return filtered_df


def main(args):

    check_args(args)

    # We first check if there is the file
    subprocess.run(
        [sys.executable, "-m",  "scripts.check_data", "--root", str(args.root), "--only", "ego4d-annotations"],
        check=True
    )

    file_path = args.root / "ego4d" / "annotations" / "egoclip.csv"

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(
        filepath_or_buffer=file_path,
        sep="\t",
        on_bad_lines="skip", # There are some errors in the csv, thus this is essential
        usecols=range(10), # The 11th col is garbage
    )

    cleaned_df = clean_df(df=df, args=args)

    # We create a new dataframe and here we count how many samples (narrations) there are for every video
    # Then we apply a cumsum wrt the 'count' column so that we know how many videos we need to satisfy our need.
    # Example. Suppose that we want 50 samples and we have a dataframe like this:
    # video_uid | count | cumsum | satisfy_condition
    #     1         20      20       False
    #     2         30      50       True
    #     3         10      60       True
    # We know that, in order to have 50 samples, we need to cut at the second row (hence, we need 2 videos)
    counts = cleaned_df["video_uid"].value_counts()
    n_videos_needed = int((counts.cumsum() < args.nSamples).sum()) + 1
    uid_videos_to_be_downloaded = counts.iloc[:n_videos_needed].index.tolist()

    benchmark_df = (
        cleaned_df[cleaned_df["video_uid"].isin(uid_videos_to_be_downloaded)]
        .head(args.nSamples)
    )

    out_dir = args.root / "ego4d" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    uids_file = out_dir / "video_uids_to_download.txt"
    clips_file = out_dir / "ego4d_benchmark_clips.csv"

    uids_file.write_text("\n".join(uid_videos_to_be_downloaded) + "\n")
    benchmark_df.to_csv(clips_file, sep="\t", index=False)

    print(f"Selected {len(benchmark_df)} clips from {len(uid_videos_to_be_downloaded)} videos.")
    print(f"List UID saved in: {uids_file}")
    print(f"Metadata clip saved in: {clips_file}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
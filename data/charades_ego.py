"""
Charades-Ego dataset for zero-shot (multi-label) action recognition.
"""

import logging
from os import path as osp

import decord
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch.nn import functional as F
from torch.utils.data import Dataset

decord.bridge.set_bridge("torch")

logger = logging.getLogger(__name__)


class CharadesEgoDataset(Dataset):
    """Charades-Ego, first-person videos only (`CharadesEgo_v1_<split>_only1st.csv`).

    Each sample is `(video_id, frames, labels)`: `frames` are `num_frames` frames uniformly sampled
    from the raw video (decoded with decord) and passed through `transforms`; `labels` is the
    multi-hot vector over the 157 Charades action classes. Videos without any annotated action are
    skipped.
    """

    def __init__(
        self,
        anns_root: str = "grace_data/charades_ego/annotations",
        videos_root: str = "grace_data/charades_ego/videos",
        split: str = "test",
        transforms=None,
        num_frames: int = 16,
        **kwargs,
    ):
        """
        Parameters
        ----------
        anns_root : str
            Directory with the Charades-Ego annotation csv files and `Charades_v1_classes.txt`.
        videos_root : str
            Directory with the `<video_id>.mp4` videos.
        split : str
            Either "train" or "test".
        transforms : callable, optional
            Applied to the `(T, C, H, W)` uint8 frames of each video.
        num_frames : int
            Number of frames uniformly sampled from each video.
        """
        self.videos_root = videos_root
        self.transforms = transforms
        self.num_frames = num_frames

        # Each line is "c<idx> <label>", e.g. "c000 Holding some clothes".
        with open(osp.join(anns_root, "Charades_v1_classes.txt"), encoding="utf-8") as f:
            self.actions_labels = [line.rstrip("\n")[5:] for line in f]

        annotations = pd.read_csv(osp.join(anns_root, f"CharadesEgo_v1_{split}_only1st.csv"))
        self.data: list[tuple[str, torch.Tensor]] = []
        for _, row in annotations.iterrows():
            if pd.isna(row["actions"]):
                continue
            # Each action is "c<idx> <start> <end>"; only the class matters for recognition.
            actions = [int(action.split(" ")[0][1:]) for action in row["actions"].split(";")]
            labels = F.one_hot(torch.tensor(actions), num_classes=len(self.actions_labels)).sum(dim=0).clamp(max=1)
            self.data.append((row["id"], labels))

        logger.info("Loaded %d samples for the Charades-Ego %s split.", len(self.data), split)

    def labels(self) -> list[str]:
        """Names of the action classes, in label-index order."""
        return self.actions_labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        video_id, labels = self.data[idx]
        frames = self.load_frames(osp.join(self.videos_root, f"{video_id}.mp4"))
        return video_id, frames, labels

    def load_frames(self, video_path: str) -> torch.Tensor:
        """Decode `num_frames` uniformly sampled frames from a video."""
        video_reader = decord.VideoReader(video_path, num_threads=1)
        indices = sample_frames(self.num_frames, len(video_reader))
        frames = rearrange(video_reader.get_batch(indices), "T H W C -> T C H W")
        if self.transforms:
            frames = self.transforms(frames)
        return frames


def sample_frames(num_frames: int, vlen: int) -> list[int]:
    """Split the video into `num_frames` equal segments and pick the middle frame index of each."""
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = [(interv, intervals[idx + 1] - 1) for idx, interv in enumerate(intervals[:-1])]
    return [(x[0] + x[1]) // 2 for x in ranges]

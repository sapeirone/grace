"""EgoClip dataset."""

import json
import logging
import os
import os.path as osp
from collections import namedtuple
from math import ceil, floor

import decord
import einops
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset
from tqdm.auto import tqdm

decord.bridge.set_bridge("torch")

logger = logging.getLogger(__name__)

Sample = namedtuple("Sample", ["video_uid", "narration_source", "narration_ind", "narration", "start_frame", "end_frame", "tags"])


class EgoClipDataset(Dataset):
    """Dataset for EgoClip, the 3.8M-narration training split of Ego4D used to train GRACE."""

    def __init__(self, anns_root: str, features_root: str, videos_root: str, ret_videos: bool, num_frames: int, transforms: T.Compose, **kwargs):
        """Initialize the EgoClip dataset.

        Parameters
        ----------
        anns_root : str
            Directory containing the `egoclip.csv` narration annotations and the `ego4d.json`
            video metadata (used to look up each video's scenario tags).
        features_root : str
            Directory containing the precomputed backbone features (`features.pt`, see
            `scripts/extract_ego4d_features.py`). Only used when `ret_videos` is False.
        videos_root : str
            Directory containing the raw `<video_uid>.mp4` files. Only used when `ret_videos`
            is True.
        ret_videos : bool
            If True, decode raw frames from `videos_root` via decord instead of loading
            precomputed features from `features_root`.
        num_frames : int
            Number of frames to sample per clip when `ret_videos` is True.
        transforms : T.Compose
            Preprocessing transform applied to the decoded frames. Only used when `ret_videos`
            is True.
        """
        self.features_root = features_root
        self.videos_root = videos_root

        self.ret_videos = ret_videos
        self.num_frames = num_frames

        if self.ret_videos:
            logger.info("Checking if the videos root is valid...")
            assert self.videos_root is not None, "videos_root must be provided if ret_videos is True"
            assert osp.exists(self.videos_root) and len(os.listdir(self.videos_root)) > 0, "videos_root is empty or does not exist"
            logger.info("Checking if the videos root is valid... Done!")

            self.transforms = transforms
        else:
            self.features = torch.load(osp.join(features_root, "features.pt"), weights_only=True, map_location="cpu", mmap=True)

        # Load samples
        self.samples = self._load_samples(anns_root)

    def _load_samples(self, anns_root: str):
        """Parse `egoclip.csv` into per-narration samples, skipping ones without usable
        features.

        Parameters
        ----------
        anns_root : str
            Directory containing the `egoclip.csv` narration annotations and the
            `ego4d.json` video metadata.

        Returns
        -------
        list of Sample
            One `Sample` namedtuple per sample, where `tags` is a one-hot vector over the
            video's Ego4D scenario labels.
        """
        samples = []

        logger.info("Initializing the EgoClip annotations...")

        annotations = pd.read_csv(f"{anns_root}/egoclip.csv", sep="\t", on_bad_lines="skip")
        annotations = annotations.sort_values(by=["video_uid", "narration_ind"])

        # Extract unique scenario labels from the ego4d.json annotations
        ego4d_video_annotations = json.load(open(f"{anns_root}/ego4d.json", encoding="utf-8"))
        scenarios_mapping = {video["video_uid"]: video["scenarios"] for video in ego4d_video_annotations["videos"]}
        unique_scenarios = sorted({scenario for scenarios in scenarios_mapping.values() for scenario in scenarios})

        # Videos with precomputed features (the keys of features.pt), to skip missing ones
        existing_features = set() if self.ret_videos else set(self.features.keys())

        skipped_videos = set()

        for _, sample in tqdm(annotations.iterrows(), desc="Loading EgoClip annotations...", leave=False, total=len(annotations)):
            video_start_frame = int(floor(sample.clip_start * 30.0))
            video_end_frame = int(ceil(sample.clip_end * 30.0))

            video_uid = sample.video_uid
            narration_source = sample.narration_source
            narration_ind = sample.narration_ind

            narration = sample.clip_text.strip().lower()

            if not self.ret_videos and video_uid not in existing_features:
                # skip samples for which we do not have the visual features
                skipped_videos.add(video_uid)
                continue

            sample_scenarios = scenarios_mapping.get(video_uid, [])
            if len(sample_scenarios) == 0:
                tags = torch.zeros(len(unique_scenarios), dtype=torch.long)
            else:
                tags = torch.nn.functional.one_hot(torch.LongTensor([unique_scenarios.index(s) for s in sample_scenarios]), num_classes=len(unique_scenarios)).max(0).values  # type: ignore

            samples.append(Sample(video_uid, narration_source, narration_ind, narration, video_start_frame, video_end_frame, tags))

        if len(skipped_videos) > 0:
            logger.warning("Skipped samples from %d videos (missing features).", len(skipped_videos))

        logger.info("Initializing the EgoClip annotations... Done!")
        return samples

    def __len__(self):
        """Number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample, as raw frames or precomputed features depending on `self.ret_videos`.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple
            See `_getitem_frames` / `_getitem_features` for the exact tuple contents.
        """
        if self.ret_videos:
            return self._getitem_frames(idx)
        return self._getitem_features(idx)

    def _getitem_frames(self, idx):
        """Decode and preprocess the raw video frames for sample `idx`.

        Returns
        -------
        tuple
            `(frames, narration, metadata)`, where `frames` has shape
            `(num_frames, C, H, W)` after `self.transforms`.
        """
        sample = self.samples[idx]
        sample_uid = f"{sample.narration_source}_{sample.narration_ind}"
        metadata = {"video_uid": sample.video_uid, "start_frame": sample.start_frame, "end_frame": sample.end_frame, "sample_uid": sample_uid}

        logger.debug("Loading raw frames for sample %s (sf=%d, ef=%d, num_frames=%d)...", sample.video_uid, sample.start_frame, sample.end_frame, self.num_frames)
        video_path = osp.join(self.videos_root, f"{sample.video_uid}.mp4")
        frames = load_frames(video_path, sample.start_frame, sample.end_frame, num_frames=self.num_frames)
        frames = self.transforms(frames)

        return frames, sample.narration, metadata

    def _getitem_features(self, idx):
        """Look up the precomputed backbone features for sample `idx`.

        Returns
        -------
        tuple
            `(video_features, text_features, narration, metadata)`. `metadata` includes
            the sample's Ego4D scenario `tags`.
        """
        sample = self.samples[idx]
        sample_uid = f"{sample.narration_source}_{sample.narration_ind}"
        metadata = {"video_uid": sample.video_uid, "start_frame": sample.start_frame, "end_frame": sample.end_frame, "sample_uid": sample_uid, "id": idx, "tags": sample.tags}

        features = self.features[sample.video_uid][sample.narration_source]
        video_features, text_features = features["video_features"][sample.narration_ind], features["text_features"][sample.narration_ind]
        video_features, text_features = video_features.float(), text_features.float()

        return video_features, text_features, sample.narration, metadata


def load_frames(video_path: str, start_frame: int, end_frame: int, num_frames: int, annotation_fps: float = 30.0) -> torch.Tensor:
    """Decode `num_frames` frames evenly sampled from `[start_frame, end_frame)` of a video.

    Parameters
    ----------
    video_path : str
        Path to the video file to decode.
    start_frame : int
        First frame index of the clip (inclusive), at `annotation_fps`.
    end_frame : int
        Last frame index of the clip (exclusive), at `annotation_fps`.
    num_frames : int
        Number of frames to sample from the clip.
    annotation_fps : float, optional
        Frame rate of `start_frame` / `end_frame`, by default 30 (the Ego4D videos). A video at another
        frame rate (e.g. a subsampled copy) gets the frames nearest in time to the same indices.

    Returns
    -------
    torch.Tensor
        Decoded frames, shaped `(num_frames, C, H, W)`.
    """
    video_reader = decord.VideoReader(video_path, num_threads=1)
    fps = video_reader.get_avg_fps()

    if abs(fps - annotation_fps) < 0.5:
        indices = np.linspace(start_frame, min(len(video_reader) - 1, end_frame - 1), num=num_frames, dtype=int)
    else:
        num_annotation_frames = round(len(video_reader) * annotation_fps / fps)
        indices = np.linspace(start_frame, min(num_annotation_frames - 1, end_frame - 1), num=num_frames, dtype=int)
        indices = np.clip(np.round(indices * fps / annotation_fps).astype(int), 0, len(video_reader) - 1)
    frames = video_reader.get_batch(indices)

    return einops.rearrange(frames, "T H W C -> T C H W")

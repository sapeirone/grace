"""EgoMCQ dataset."""

import json
import logging
import os
import os.path as osp
from collections import namedtuple
from math import ceil, floor

import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

from data.egoclip import load_frames

logger = logging.getLogger(__name__)

Sample = namedtuple("Sample", ["ann_id", "query_text", "choices", "answer", "types", "query_video_uid"])


class EgoMCQDataset(Dataset):
    """EgoMCQ multiple-choice benchmark (Ego4D): given a narration, pick its matching video clip
    among a set of candidate choices."""

    def __init__(self, anns_root: str, features_root: str, videos_root: str, ret_videos: bool, transforms: T.Compose, num_frames: int = 16, **kwargs):
        """Initialize the EgoMCQ dataset.

        Parameters
        ----------
        anns_root : str
            Directory containing the `egomcq.json` annotations.
        features_root : str
            Directory containing precomputed backbone features (`features.pt`, one entry per
            question). Only used when `ret_videos` is False.
        videos_root : str
            Directory containing the raw `<video_uid>.mp4` files. Only used when `ret_videos`
            is True.
        ret_videos : bool
            If True, decode raw frames for each candidate choice from `videos_root` via decord
            instead of loading precomputed features from `features_root`.
        transforms : T.Compose
            Preprocessing transform applied to the decoded frames. Only used when `ret_videos`
            is True.
        num_frames : int, optional
            Number of frames to sample per candidate choice when `ret_videos` is True, by
            default 16.
        """
        self.features_root = features_root
        self.videos_root = videos_root

        self.ret_videos = ret_videos
        self.num_frames = num_frames

        self.transforms = transforms

        if self.ret_videos:
            logger.info("Checking if the videos root is valid...")
            assert self.videos_root is not None, "videos_root must be provided if ret_videos is True"
            assert osp.exists(self.videos_root) and len(os.listdir(self.videos_root)) > 0, "videos_root is empty or does not exist"
            logger.info("Checking if the videos root is valid... Done!")
        else:
            self.features = torch.load(osp.join(features_root, "features.pt"), weights_only=True, map_location="cpu")
            # Feature size, for the zero features of the questions missing from `features.pt`
            self.features_size = next(iter(self.features.values()))[1].shape[-1]

        # Load samples
        self.samples = self._load_samples(anns_root)

    def _load_samples(self, anns_root: str):
        """Parse `egomcq.json` into per-question samples.

        Parameters
        ----------
        anns_root : str
            Directory containing the `egomcq.json` annotations.

        Returns
        -------
        list of Sample
            One `Sample` namedtuple per question, where `choices` is a list of
            `(video_uid, start_frame, end_frame)` for each candidate clip.
        """
        annotations = json.load(open(osp.join(anns_root, "egomcq.json")))

        samples = []
        for ann_id, sample in annotations.items():
            query = sample["query"]
            query_video_uid = query["video_uid"]

            choices = []
            for choice in sample["choices"].values():
                start, end = floor(float(choice["clip_start"]) * 30.0), ceil(float(choice["clip_end"]) * 30.0)
                choices.append((choice["video_uid"], start, end))

            samples.append(Sample(str(ann_id), query["clip_text"].strip().lower(), choices, sample["answer"], sample["types"], query_video_uid))

        return samples

    def __len__(self) -> int:
        """Number of questions in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a question, as raw frames or precomputed features depending on `self.ret_videos`.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple
            See `_getitem_frames` / `_getitem_features` for the exact tuple contents.
            In both cases, `metadata` carries `answer` (the index of the correct choice
            among `choices`, or -1 if features are missing) and the question `type`
            (1 = inter-video, 2 = intra-video).
        """
        if self.ret_videos:
            return self._getitem_frames(idx)
        return self._getitem_features(idx)

    def _getitem_frames(self, idx):
        """Decode and preprocess the raw video frames for each candidate choice of sample `idx`.

        Returns
        -------
        tuple
            `(frames, text, metadata)`, where `frames` has shape `(num_choices,
            num_frames, C, H, W)` after `self.transforms`.
        """
        sample = self.samples[idx]

        logger.debug("Loading raw frames for sample %s...", sample.ann_id)
        frames = torch.vstack([self._load_frames(vid, start, end)[None] for vid, start, end in sample.choices])  # (num_choices, num_frames, C, H, W)
        return frames, sample.query_text, {"video_uid": sample.ann_id, "answer": sample.answer, "type": sample.types}

    def _getitem_features(self, idx):
        """Look up the precomputed backbone features for sample `idx`.

        Returns
        -------
        tuple
            `(video_features, text_features, text, metadata)`. `metadata['answer']` is
            -1 if the sample has no precomputed features (see `__getitem__`).
        """
        sample = self.samples[idx]

        answer = sample.answer
        if str(sample.ann_id) not in self.features:
            # Invalid sample: a very small number of videos is absent since these features were
            # extracted from Ego4Dv2, which dropped some very small videos.
            video_features = torch.zeros((len(sample.choices), self.features_size), dtype=torch.float32)
            text_features = torch.zeros(self.features_size, dtype=torch.float32)
            answer = -1
        else:
            video_features, text_features = self.features[str(sample.ann_id)]

        video_features, text_features = torch.clone(video_features.float()), torch.clone(text_features.float())

        metadata = {"video_uid": sample.ann_id, "query_video_uid": sample.query_video_uid, "answer": answer, "type": sample.types}

        return video_features, text_features, sample.query_text, metadata

    def _load_frames(self, video_uid: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """Decode and preprocess the raw frames for one candidate choice."""
        video_path = osp.join(self.videos_root, f"{video_uid}.mp4")
        frames = load_frames(video_path, start_frame, end_frame, num_frames=self.num_frames)
        return self.transforms(frames)

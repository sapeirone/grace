import json
import os

import numpy as np
from decord import VideoReader, cpu
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """The videos of the ConViS-Bench pairs, each decoded to `num_frames` uniformly sampled frames."""

    def __init__(self, video_dir: str, pairs_jsonl: str, num_frames: int = 16):
        self.num_frames = num_frames
        videos = set()
        with open(pairs_jsonl, "r") as f:
            for line in f:
                pair = json.loads(line)
                videos.update([pair["video1"], pair["video2"]])
        self.video_list = [os.path.join(video_dir, video) for video in videos]

    def __len__(self):
        return len(self.video_list)

    def _load_video(self, video_path: str) -> np.ndarray:
        video_reader = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(video_reader)
        if total_frames >= self.num_frames:
            return video_reader.get_batch(np.linspace(0, total_frames - 1, self.num_frames, dtype=int)).asnumpy()
        # Shorter videos: all their frames, padded by repeating the last one
        frames = video_reader.get_batch(np.arange(total_frames)).asnumpy()
        padding = np.repeat(frames[-1:], self.num_frames - total_frames, axis=0)
        return np.concatenate([frames, padding], axis=0)

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        return {
            "video_id": video_path.split("/")[-1].split(".")[0],
            "video_path": video_path,
            "video_frames": self._load_video(video_path),
        }

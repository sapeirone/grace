"""
Precomputed backbone features of the ImageNet training set, for training GRACE on images.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)


def load_imagenet_features(features_path: str, root: str, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
    """Load precomputed ImageNet features and their labels.

    Parameters
    ----------
    features_path : str
        `torch.save`d dict mapping each sample's index in `torchvision.datasets.ImageFolder(root/split)`
        order to its `(D,)` feature, as produced by `scripts/extract_imagenet_features.py` (CLIP ViT-B/32
        with the val transform).
    root : str
        ImageNet root (with one directory per class under `root/split`), used for the labels.
    split : str
        Split directory under `root`, by default "train".

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        `(N, D)` float32 features and `(N,)` int64 labels, in ImageFolder order.
    """
    logger.info("Loading ImageNet features from %s...", features_path)
    features_dict = torch.load(features_path, map_location="cpu", weights_only=False, mmap=True)
    features = torch.stack([features_dict[i] for i in range(len(features_dict))]).float()

    # ImageFolder order: classes sorted by directory name, files sorted within each class (only image
    # files; ImageNet class directories contain nothing else).
    split_dir = os.path.join(root, split)
    classes = sorted(entry.name for entry in os.scandir(split_dir) if entry.is_dir())
    counts = [sum(1 for entry in os.scandir(os.path.join(split_dir, c)) if entry.is_file()) for c in classes]
    labels = torch.repeat_interleave(torch.arange(len(classes)), torch.tensor(counts))

    assert len(labels) == len(features), f"{len(features)} features but {len(labels)} images under {split_dir}"
    logger.info("Loaded %d features of size %d (%d classes)", len(features), features.shape[1], len(classes))
    return features, labels

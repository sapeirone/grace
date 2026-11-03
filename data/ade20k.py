"""
ADE20k (SceneParsing / ADEChallengeData2016) images with patch-level semantic labels, for the
concept localization eval.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ADE20KDataset(Dataset):
    """ADE20k images, each with the semantic label of every patch of a `grid x grid` patch grid.

    Each sample is `(image, patch_labels, filename)`: `image` is the RGB image passed through
    `transforms`; `patch_labels` is a `(grid, grid)` int64 tensor with the most frequent label of
    each patch (0 = "other", 1..150 = the classes of `objectInfo150.txt`, ties broken towards the
    lowest label), computed on the annotation resized (nearest neighbour) to
    `grid * patch_size` pixels per side.
    """

    def __init__(self, root: str, split: str = "validation", transforms=None, grid: int = 32, patch_size: int = 14, **kwargs):
        """
        Parameters
        ----------
        root : str
            The `ADEChallengeData2016` directory (with `images/`, `annotations/`, `objectInfo150.txt`).
        split : str
            Either "training" or "validation".
        transforms : callable, optional
            Applied to each PIL image.
        grid : int
            Number of patches per side, by default 32 (448px images with a /14 backbone).
        patch_size : int
            Side of a patch in pixels, by default 14.
        """
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.grid = grid
        self.patch_size = patch_size

        self.files = sorted(os.listdir(self.root / "images" / split))
        # Skip the header ("Idx Ratio Train Val Name"); the class name is the last column.
        self.classes = [line.split("\t")[-1] for line in (self.root / "objectInfo150.txt").read_text().splitlines()[1:]]
        assert len(self.classes) == 150

        side = grid * patch_size
        self.mask_transform = T.Compose([T.ToImage(), T.Resize((side, side), interpolation=T.InterpolationMode.NEAREST)])

        logger.info("Loaded %d images for the ADE20k %s split.", len(self.files), split)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        fn = self.files[idx]
        image = Image.open(self.root / "images" / self.split / fn).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        annotation = Image.open(self.root / "annotations" / self.split / fn.replace(".jpg", ".png"))
        return image, torch.from_numpy(self.patch_labels(annotation)), fn

    def patch_labels(self, annotation: Image.Image) -> np.ndarray:
        """Most frequent label of each patch of the (resized) annotation, as a `(grid, grid)` array."""
        label_map = np.array(self.mask_transform(annotation).squeeze(0))
        g, p = self.grid, self.patch_size
        patches = label_map.reshape(g, p, g, p).transpose(0, 2, 1, 3).reshape(g * g, p * p)
        counts = np.zeros((g * g, len(self.classes) + 1), dtype=np.int32)
        np.add.at(counts, (np.arange(g * g)[:, None].repeat(p * p, axis=1), patches), 1)
        return counts.argmax(axis=1).reshape(g, g).astype(np.int64)

    def labels(self) -> list[str]:
        """Names of the 150 classes; class `i` has label `i + 1` in the annotations."""
        return self.classes

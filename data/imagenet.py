"""ImageNet dataset, used for zero-shot classification."""

from pathlib import Path

from torchvision.datasets import ImageFolder


class ImageNetDataset(ImageFolder):
    """ImageNet-1k, loaded from a local `ImageFolder`-compatible directory layout."""

    def __init__(self, root: str | Path, split: str, **kwargs):
        """Initialize the ImageNet dataset.

        Parameters
        ----------
        root : Union[str, Path]
            Directory containing one subdirectory per split (e.g. `root/train`, `root/val`),
            each in `torchvision.datasets.ImageFolder` layout (one subdirectory per class).
        split : str
            Split to load (e.g. "train" or "val").
        """
        super().__init__(Path(root) / split, **kwargs)

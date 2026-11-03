"""COCO Captions dataset, used for image-text retrieval."""

from pathlib import Path

from torchvision.datasets import CocoCaptions


class CocoDataset(CocoCaptions):
    """COCO 2017 Captions, loaded from the official `train2017`/`val2017` image directories and
    `captions_{split}2017.json` annotation files."""

    def __init__(self, root: str | Path, split: str, *args, **kwargs):
        """Initialize the COCO Captions dataset.

        Parameters
        ----------
        root : Union[str, Path]
            Directory containing `{split}2017/` images and `annotations/captions_{split}2017.json`.
        split : str
            Split to load (e.g. "train" or "val").
        """
        root = Path(root)

        super().__init__(
            root / f"{split}2017",
            root / "annotations" / f"captions_{split}2017.json",  # type: ignore[arg-type]
            *args,
            **kwargs,
        )

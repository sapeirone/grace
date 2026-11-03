"""Flickr30k dataset, used for image-text retrieval."""

from pathlib import Path

from torchvision.datasets import Flickr30k


class Flickr30kDataset(Flickr30k):
    """Flickr30k, loaded from an `Images/` directory and a `flickr30k_{split}.tsv` annotation file."""

    def __init__(self, root: str | Path, split: str, *args, **kwargs):
        """Initialize the Flickr30k dataset.

        Parameters
        ----------
        root : Union[str, Path]
            Directory containing `Images/` and `flickr30k_{split}.tsv`.
        split : str
            Split to load (e.g. "train" or "test").
        """
        root = Path(root)

        super().__init__(root / "Images", root / f"flickr30k_{split}.tsv", *args, **kwargs)  # type: ignore[arg-type]

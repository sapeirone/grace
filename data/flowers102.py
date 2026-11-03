"""Flowers102 dataset, used for zero-shot classification."""

from torchvision.datasets import Flowers102


class Flowers102Dataset(Flowers102):
    """Flowers102, loaded via its official train/val/test split (`torchvision.datasets.Flowers102`
    downloads and splits the dataset itself)."""

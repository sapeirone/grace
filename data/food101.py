"""Food101 dataset, used for zero-shot classification."""

from torchvision.datasets import Food101


class Food101Dataset(Food101):
    """Food101, loaded via its official train/test split (`torchvision.datasets.Food101`
    downloads and splits the dataset itself)."""

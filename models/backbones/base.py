"""This file defines the base class for all CLIP models in this project."""

import torch
import torchvision.transforms.v2 as T
from torch import nn


class CLIP(nn.Module):
    """Base class for all CLIP-like models in this project."""

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Compute features for the input images.

        Parameters
        ----------
        images : torch.Tensor
            input images

        Returns
        -------
        torch.Tensor
            features extracted from the raw images
        """
        raise NotImplementedError()

    def encode_text(self, text: list[str]) -> torch.Tensor:
        """Compute features for the input text.

        Parameters
        ----------
        text : List[str]
            input text

        Returns
        -------
        torch.Tensor
            features extracted from the input text
        """
        raise NotImplementedError()

    @property
    def features_size(self) -> int:
        """Get the size of the features extracted by the model.

        Returns
        -------
        int
            size of the features extracted by the model
        """
        raise NotImplementedError()

    @property
    def context_length(self) -> int | None:
        """Get the context size of the text model.

        Returns
        -------
        int
            context size of the text model
        """
        return None

    @property
    def preprocess(self) -> T.Compose:
        """Get the preprocessing transform for the model.

        Returns
        -------
        T.Compose
            preprocessing transform
        """
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        """Get the device where the model is located.

        Returns
        -------
        torch.device
            device where the model is located
        """
        return next(self.parameters()).device

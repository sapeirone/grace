"""This file defines the openai/CLIP backbone, used for the image-side experiments."""

import logging

import clip
import torch

from data.transforms import get_clip_val_image_transform
from models.backbones.base import CLIP as CLIPBase

logger = logging.getLogger(__name__)


class CLIPBackbone(CLIPBase):
    """Wrapper for the openai/CLIP backbone."""

    def __init__(self, name: str = "clip", variant: str = "ViT-B/32", **kwargs):
        """Initialize the CLIP backbone.

        Parameters
        ----------
        name : str, optional
            Name of the backbone, by default 'clip'
        variant : str, optional
            Variant of the CLIP model (e.g. "ViT-B/32", "ViT-L/14"), by default "ViT-B/32"
        """
        super().__init__(name, **kwargs)

        # clip.load's own (bicubic) transform is discarded: the released checkpoints use a bilinear resize
        self.clip_model, _ = clip.load(variant, device="cpu")
        self._preprocess = get_clip_val_image_transform(self.clip_model.visual.input_resolution)

        # Disable gradients for all parameters
        for param in self.parameters():
            param.requires_grad = False

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
        return self.clip_model.encode_image(images).float()

    def encode_text(self, text: list[str] | torch.Tensor) -> torch.Tensor:
        """Compute features for the input text.

        Parameters
        ----------
        text : Union[List[str], torch.Tensor]
            Input text, either as raw strings (tokenized here) or as an already-tokenized
            token-id tensor (e.g. produced upstream by a dataset's caption transform).

        Returns
        -------
        torch.Tensor
            features extracted from the input text
        """
        tokens = text.to(self.device) if torch.is_tensor(text) else clip.tokenize(text, context_length=self.context_length, truncate=True).to(self.device)
        return self.clip_model.encode_text(tokens).float()

    @property
    def features_size(self) -> int:
        """Get the size of the features extracted by the model.

        Returns
        -------
        int
            size of the features extracted by the model
        """
        return self.clip_model.visual.output_dim

    @property
    def preprocess(self):
        """Get the preprocessing transform for the model.

        Returns
        -------
        Compose
            preprocessing transform, `data.transforms.get_clip_val_image_transform` at the model's input
            resolution (bilinear resize, unlike `clip.load`'s bicubic one)
        """
        return self._preprocess

    @property
    def context_length(self) -> int | None:
        """Get the maximum context length of the text model.

        Returns
        -------
        int
            maximum context length of the text model
        """
        return self.clip_model.context_length

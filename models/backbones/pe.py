"""Perception Encoder (PE-Core) backbone, the video backbone of GRACE."""

import logging

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import torch
import torchvision.transforms.v2 as T

from models.backbones.base import CLIP

logger = logging.getLogger(__name__)


class PEBackbone(CLIP):
    """Wrapper for the PE backbone."""

    def __init__(self, name: str = "pe", variant: str = "PE-Core-L14-336", **kwargs):
        """Initialize the PerceptionEncoder backbone.

        Parameters
        ----------
        name : str, optional
            Name of the backbone, by default 'pe'
        variant : str, optional
            Variant of the PerceptionEncoder model, by default "PE-Core-L14-336"
        """
        super().__init__(name, **kwargs)

        self.pe_model = pe.CLIP.from_config(variant, pretrained=True)
        self.tokenizer = transforms.get_text_tokenizer(self.pe_model.context_length)

        # Disabled gradients for all parameters
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
            features extracted from the raw frames
        """

        return self.pe_model.encode_video(images)

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
        tokens = self.tokenizer(text).to(self.device)
        tokens = tokens[:, : self.pe_model.context_length]  # truncate if needed
        return self.pe_model.encode_text(tokens)  # type: ignore

    @property
    def features_size(self) -> int:
        """Get the size of the features extracted by the model.

        Returns
        -------
        int
            size of the features extracted by the model
        """
        return self.pe_model.visual.output_dim

    @property
    def preprocess(self) -> T.Compose:
        """Get the preprocessing function for the model.

        Returns
        -------
        T.Compose
            preprocessing function for the model
        """
        mean, std = self.normalization_stats

        return T.Compose(
            [
                T.Lambda(lambda x: x / 255.0),
                T.Resize(size=self.pe_model.image_size),
                T.Normalize(mean=mean, std=std),
                T.CenterCrop(size=self.pe_model.image_size),
            ]
        )

    @property
    def normalization_stats(self) -> tuple[list[float], list[float]]:
        """Get the normalization statistics for the model.

        Returns
        -------
        Tuple[List[float], List[float]]
            mean and std for normalization
        """
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    @property
    def context_length(self) -> int | None:
        """Get the maximum context length of the text model.

        Returns
        -------
        int
            maximum context length of the text model
        """
        return self.pe_model.context_length

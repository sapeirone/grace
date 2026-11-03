"""This file defines the TIPSv2 backbone (google/tipsv2-*, via Hugging Face remote code)."""

import logging

import torch
from torchvision import transforms
from transformers import AutoModel

from models.backbones.base import CLIP as CLIPBase

logger = logging.getLogger(__name__)

# Revision of the Hugging Face repo (model code + weights) the released TIPSv2 GRACE checkpoint
# was evaluated with; pinned because the repo's remote code changes between revisions.
DEFAULT_REVISION = "52847a71eb02a3082c370fbcd98cd799687bd4d0"


class TIPSv2Backbone(CLIPBase):
    """Wrapper for the TIPSv2 image-text models (`google/tipsv2-b14`, `google/tipsv2-l14`, ...)."""

    def __init__(self, name: str = "tipsv2", variant: str = "tipsv2-l14", revision: str = DEFAULT_REVISION, input_size: int = 448, **kwargs):
        """Initialize the TIPSv2 backbone.

        Parameters
        ----------
        name : str, optional
            Name of the backbone, by default 'tipsv2'
        variant : str, optional
            Hugging Face model name under `google/`, by default "tipsv2-l14"
        revision : str, optional
            Hugging Face revision of the model repo, by default the one the released checkpoint
            was evaluated with
        input_size : int, optional
            Side of the (square) input images, by default 448 (a 32x32 patch grid for /14 models)
        """
        super().__init__(name, **kwargs)

        self.model = AutoModel.from_pretrained(f"google/{variant}", revision=revision, trust_remote_code=True)
        self.input_size = input_size
        # Plain resize to a square + [0, 1] scaling, no normalization (what the checkpoint was trained with).
        self._preprocess = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])

        # Disable gradients for all parameters
        for param in self.parameters():
            param.requires_grad = False

    def encode_image(self, images: torch.Tensor, return_patches: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute features for the input images.

        Parameters
        ----------
        images : torch.Tensor
            input images, `(B, 3, input_size, input_size)` in [0, 1]
        return_patches : bool, optional
            Whether to also return the patch tokens, by default False

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            the (unnormalized) CLS embedding, `(B, D)`, and, if `return_patches`, the (unnormalized)
            patch embeddings as a `(B, H, W, D)` grid
        """
        out = self.model.encode_image(images)
        cls_token = out.cls_token[:, 0, :].float()
        if not return_patches:
            return cls_token
        grid = self.input_size // self.model.config.patch_size
        return cls_token, out.patch_tokens.float().reshape(images.shape[0], grid, grid, -1)

    def encode_text(self, text: list[str]) -> torch.Tensor:
        """Compute (unnormalized) features for the input text."""
        return self.model.encode_text(list(text)).float()

    @property
    def features_size(self) -> int:
        """Get the size of the features extracted by the model."""
        return int(self.model.config.embed_dim)

    @property
    def preprocess(self):
        """Get the preprocessing transform for the model (PIL image -> tensor)."""
        return self._preprocess

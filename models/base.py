import torch
from torch import nn
from torch.nn import functional as F


class InterpretableCLIP(nn.Module):
    """Interpretable CLIP-like model with concepts bottleneck."""

    def __init__(self, features_size: int, text_concepts: list[str], concepts_embds: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.text_concepts = text_concepts
        self.concepts = nn.Embedding.from_pretrained(concepts_embds, freeze=True)

        # Mean of the visual embeddings, saved with the checkpoint: SpLiCE, and GRACE with
        # `center_before_activations`, center the visual features on it.
        self.features_size = features_size
        self.mean_visual_embds = nn.Buffer(torch.zeros(features_size), persistent=True)

    def set_visual_mean(self, mean_visual_embds: torch.Tensor) -> None:
        self.mean_visual_embds.copy_(mean_visual_embds)

    def encode_image(self, visual_features: torch.Tensor, return_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode the visual features with the interpretable model.

        Parameters
        ----------
        visual_features : torch.Tensor
            Visual features
        return_weights : bool, optional
            Optionally return reconstruction weights, by default False

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            The encoded features, or, if `return_weights` is True, a tuple of (encoded features,
            reconstruction weights).
        """
        raise NotImplementedError

    def forward(self, visual_features: torch.Tensor, tags: torch.Tensor | None = None, **kwargs) -> tuple[torch.Tensor, torch.Tensor | float, dict[str, torch.Tensor]]:
        """Encode the visual features and compute the training loss.

        Parameters
        ----------
        visual_features : torch.Tensor
            Visual features
        tags : Optional[torch.Tensor], optional
            Optional sample-level tags to be used for disentanglement losses, by default None

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | float, dict[str, torch.Tensor]]
            Encoded visual features, total loss and its components (by name)
        """
        raise NotImplementedError

    def clip_loss(self, visual_features: torch.Tensor, text_features: torch.Tensor, temperature: float) -> torch.Tensor:
        """Compute the CLIP loss between visual and text features.

        Parameters
        ----------
        visual_features : torch.Tensor
            Visual features
        text_features : torch.Tensor
            Text features

        Returns
        -------
        torch.Tensor
            CLIP loss
        """
        similarities = F.normalize(visual_features, p=2, dim=-1) @ F.normalize(text_features.detach(), dim=-1).T
        return F.cross_entropy(similarities / temperature, torch.arange(visual_features.shape[0], device=visual_features.device, dtype=torch.long))

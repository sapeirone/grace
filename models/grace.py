"""GRACE: interpretable CLIP-like model with a concepts bottleneck.

Visual features are projected onto a fixed bank of pretrained concept embeddings,
"highlighted" by a learned saliency score, and sparsified into a reconstruction
weight distribution. The visual features are then reconstructed as a weighted sum
of concept embeddings, and the whole model is trained with a mix of reconstruction,
concepts-disentanglement and regularization losses.
"""

import logging
from typing import Literal

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from models.base import InterpretableCLIP

logger = logging.getLogger(__name__)

# Above this many (batch^2 x num_concepts) elements, the concepts loss computes the pairwise JSD in
# chunks instead of materializing it (the video setting, 128^2 x 700, stays below it).
_MAX_PAIRWISE_ELEMENTS = 2**28
_JSD_CHUNK_ROWS = 32


def _jsd_rows(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between each row of `p` (n, Z) and each row of `q` (B, Z), as (n, B);
    the same formula (and epsilons) as the unchunked branch of `Grace.concepts_loss`."""
    p, q = p[:, None], q[None, :]
    mixture = (0.5 * (p + q) + 1e-8).log()
    return 0.5 * ((p * (p + 1e-8).log() - p * mixture).sum(-1) + (q * (q + 1e-8).log() - q * mixture).sum(-1))


_jsd_rows_compiled = torch.compile(_jsd_rows, dynamic=False)


def _chunked_pairwise_jsd(weights: torch.Tensor) -> torch.Tensor:
    """(B, B) pairwise JSD of `weights` (B, Z), `_JSD_CHUNK_ROWS` rows at a time. Each chunk is a fused
    compiled kernel and is recomputed in the backward pass (activation checkpointing), so memory stays
    O(chunk x B x Z) instead of O(B^2 x Z)."""
    return torch.cat([torch.utils.checkpoint.checkpoint(_jsd_rows_compiled, weights[i : i + _JSD_CHUNK_ROWS], weights, use_reentrant=False) for i in range(0, weights.shape[0], _JSD_CHUNK_ROWS)])


ProjType = Literal["linear", "linear_nobias", "mlp", "identity", "ortholinear"]
ScoringActFunc = Literal["softplus", "tanh", "sigmoid", "identity"]


class Grace(InterpretableCLIP):
    """Interpretable CLIP-like model with a concepts bottleneck and contrastive-based disentanglement."""

    def __init__(
        self,
        features_size: int,
        text_concepts: list[str],
        concepts_embds: torch.Tensor,
        hidden_size: int = 768,
        # Loss weights for concepts-based reconstruction and disentanglement
        loss_weight_concepts: float = 1.0,
        loss_weight_rec: float = 1.0,
        loss_weight_ent_reg: float = 0.0,
        # Reconstruction projection
        proj_type: ProjType = "linear",
        scoring_act_func: ScoringActFunc = "sigmoid",
        # Regularization strategy
        reg_strategy: Literal["dirichlet", "entropy", "l1"] = "dirichlet",
        # Temperatures parameters for the different losses and distributions
        visual_rec_temperature: float = 0.05,
        reconstruction_temperature: float = 0.05,
        dirichlet_alpha: float = 0.0,
        symm_jsd: bool = False,
        gram_regularization: bool = False,
        center_before_activations: bool = False,
        threshold: float = 1e-2,
        soft_gating: bool = True,
        **kwargs,
    ) -> None:
        """GRACE: a concepts bottleneck that decomposes visual embeddings into weighted text concepts.

        Parameters
        ----------
        features_size : int
            Size of the input (backbone) features.
        text_concepts : List[str]
            Human-readable names of the concepts, in the same order as `concepts_embds`.
        concepts_embds : torch.Tensor
            Pretrained concepts embeddings, shape (num_concepts, features_size).
        hidden_size : int, optional
            Hidden size of the visual highlighter MLP, by default 768
        loss_weight_concepts : float, optional
            Weight of the concepts-based disentanglement loss, by default 1.0
        loss_weight_rec : float, optional
            Weight of the (visual) reconstruction loss, by default 1.0
        loss_weight_ent_reg : float, optional
            Weight of the regularization loss (see `reg_strategy`), by default 0.0
        proj_type : ProjType, optional
            Type of projection applied to the concept embeddings before reconstruction,
            by default "linear"
        scoring_act_func : ScoringActFunc, optional
            Activation function for the highlighter scores, by default "sigmoid"
        reg_strategy : {"dirichlet", "entropy", "l1"}, optional
            Strategy used to regularize the reconstruction weights, by default "dirichlet"
        visual_rec_temperature : float, optional
            Temperature for the visual reconstruction (CLIP) loss, by default 0.05
        reconstruction_temperature : float, optional
            Temperature used when turning reconstruction weights into a distribution
            over concepts, by default 0.05
        dirichlet_alpha : float, optional
            Alpha parameter for the Dirichlet regularization, by default 0.0
        symm_jsd : bool, optional
            Whether to make the concepts disentanglement loss symmetric (i.e. also push
            unrelated samples' concepts apart), by default False
        gram_regularization : bool, optional
            Whether to add a Gram-matrix regularization term to the reconstruction loss,
            to better preserve intra-modal pairwise similarities, by default False
        center_before_activations : bool, optional
            Whether to center visual features on the dataset mean (`mean_visual_embds`)
            before computing concept activations, by default False
        threshold : float, optional
            Activation threshold below which concepts are (softly or hardly) gated out
            of the reconstruction, by default 1e-2
        soft_gating : bool, optional
            Whether to gate low-activation concepts with a smooth sigmoid-based penalty
            instead of hard-masking them to -inf, by default True
        """

        super().__init__(features_size, text_concepts, concepts_embds)

        assert scoring_act_func in ["softplus", "tanh", "sigmoid", "identity"], f"Invalid scoring activation function: {scoring_act_func}"
        assert reg_strategy in ["dirichlet", "entropy", "l1"], f"Invalid regularization strategy: {reg_strategy}"
        assert proj_type in ["linear", "linear_nobias", "ortholinear", "mlp", "identity"], f"Invalid projection type: {proj_type}"

        # --- Store the configuration ---
        self.hidden_size = hidden_size
        self.proj_type = proj_type
        self.scoring_act_func = scoring_act_func
        self.reg_strategy = reg_strategy

        # Loss weights
        self.loss_weight_concepts = loss_weight_concepts
        self.loss_weight_rec = loss_weight_rec
        self.loss_weight_ent_reg = loss_weight_ent_reg

        # Loss parameters
        self.visual_rec_temperature = visual_rec_temperature
        self.reconstruction_temperature = reconstruction_temperature
        self.dirichlet_alpha = dirichlet_alpha

        # Flags
        self.symm_jsd = symm_jsd
        self.gram_regularization = gram_regularization
        self.center_before_activations = center_before_activations
        self.soft_gating = soft_gating
        self.threshold = threshold

        # --- Log the full configuration, for reproducibility ---
        logger.info("Initializing Grace model: features_size=%d, num_concepts=%d, hidden_size=%d", self.features_size, len(self.concepts.weight), self.hidden_size)
        logger.info("Loss weights: concepts=%.4f, rec=%.4f, ent_reg=%.4f", self.loss_weight_concepts, self.loss_weight_rec, self.loss_weight_ent_reg)
        logger.info("Loss temperatures: visual_rec=%.4f, reconstruction=%.4f", self.visual_rec_temperature, self.reconstruction_temperature)
        logger.info("Reconstruction: proj_type=%s, scoring_act_func=%s, reg_strategy=%s, dirichlet_alpha=%.4f", self.proj_type, self.scoring_act_func, self.reg_strategy, self.dirichlet_alpha)
        logger.info("Gating: soft_gating=%s, threshold=%.4f", self.soft_gating, self.threshold)
        logger.info("Flags: symm_jsd=%s, gram_regularization=%s, center_before_activations=%s", self.symm_jsd, self.gram_regularization, self.center_before_activations)
        if kwargs:
            logger.info("Additional (unrecognized) keyword arguments: %s", kwargs)

        # --- Build the model's submodules ---
        self.visual_highlighter_proj = self._build_highlighter(self.scoring_act_func)
        # Learnable mapping applied to the (frozen) concept embeddings before they are
        # used to reconstruct the visual features.
        self.concepts_proj = self._build_reconstruction_projection(self.proj_type, self.features_size)

    def _build_highlighter(self, act_func: ScoringActFunc) -> nn.Module:
        """Build the highlighter module.

        The highlighter is an MLP, followed by `act_func`, that maps visual features to
        one saliency score per concept. Concept activations are multiplied by these
        scores before reconstruction, letting the model attenuate or amplify concepts
        that are only weakly relevant to a given sample.

        Parameters
        ----------
        act_func : ScoringActFunc
            Activation function applied to the output scores.

        Returns
        -------
        nn.Module
            The highlighter module.
        """
        acts = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "softplus": nn.Softplus, "identity": nn.Identity}

        return nn.Sequential(
            nn.Linear(self.features_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, len(self.concepts.weight)),
            acts[act_func](),
        )

    @staticmethod
    def _build_reconstruction_projection(proj_type: ProjType, backbone_size: int, hidden_size: int = 1024) -> nn.Module:
        """Build the projection applied to concept embeddings before reconstruction.

        This lets the (frozen) concept embeddings be remapped into a space better
        suited for reconstructing the backbone's visual features.

        Parameters
        ----------
        proj_type : ProjType
            Projection type: "linear", "linear_nobias", "ortholinear", "mlp" or "identity".
        backbone_size : int
            Size of the backbone's features (input and output size of the projection).
        hidden_size : int, optional
            Hidden size of the "mlp" projection, by default 1024

        Returns
        -------
        nn.Module
            The reconstruction projection.
        """
        if proj_type == "linear" or proj_type == "linear_nobias":
            return nn.Linear(backbone_size, backbone_size, bias=(proj_type == "linear"))

        if proj_type == "ortholinear":
            # Orthogonally-parametrized linear layer (no bias), so the projection
            # preserves norms and pairwise distances between concept embeddings.
            return torch.nn.utils.parametrizations.orthogonal(nn.Linear(backbone_size, backbone_size, bias=False))

        if proj_type == "mlp":
            return nn.Sequential(
                nn.Linear(backbone_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, backbone_size),
            )

        # "identity": no projection.
        return nn.Identity()

    def _reconstruct(self, rec_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn raw (highlighted) concept activations into a sparse reconstruction.

        The raw activations are gated at `self.threshold` (softly or hardly, see `soft_gating`) and
        turned into a probability distribution over concepts with a softmax, which is
        then used to reconstruct the input features as a weighted sum of concept
        embeddings.

        Parameters
        ----------
        rec_weights : torch.Tensor
            Raw (highlighted) concept activations, shape (B, num_concepts).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The reconstruction weights (a distribution over concepts, sums to 1 along
            the last dimension) and the reconstructed features.
        """
        concepts = self.concepts_proj(self.concepts.weight.detach())  # type: ignore

        if self.soft_gating:
            # Soft gating: a steep sigmoid around `threshold` smoothly pushes
            # sub-threshold activations towards -inf, encouraging sparsity while
            # remaining differentiable everywhere.
            rec_weights = rec_weights + torch.log(torch.sigmoid(100 * (rec_weights - self.threshold)) + 1e-12)
        else:
            # Hard gating: activations below the threshold are dropped entirely.
            rec_weights = rec_weights.masked_fill(rec_weights < self.threshold, -float("inf"))

        # Temperature-scaled softmax turns the gated activations into a distribution
        # over concepts.
        rec_weights = (rec_weights / self.reconstruction_temperature).softmax(-1)

        # Reconstruct the original features as a weighted sum of (normalized) concepts
        return rec_weights, rec_weights @ F.normalize(concepts, dim=-1)

    def encode_image(self, visual_features: torch.Tensor, return_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode visual features through the concepts bottleneck.

        Visual features are matched against the concept bank via cosine similarity,
        the resulting activations are rescaled by a per-sample, per-concept saliency
        score from the highlighter, and the highlighted activations are sparsified and
        used to reconstruct the visual features as a weighted sum of concept embeddings.

        Parameters
        ----------
        visual_features : torch.Tensor
            Input visual features, shape (B, features_size).
        return_weights : bool, optional
            Whether to also return the reconstruction weights, by default False

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            The reconstructed visual features, shape (B, features_size), or, if `return_weights`
            is True, a tuple of (reconstructed visual features, reconstruction weights of shape
            (B, num_concepts)).
        """
        # Cosine similarity between visual features and (frozen) concept embeddings
        if self.center_before_activations:
            centered_visual_features = F.normalize(visual_features, p=2, dim=-1) - F.normalize(self.mean_visual_embds, p=2, dim=-1)
            activations = F.normalize(centered_visual_features, dim=-1) @ F.normalize(self.concepts.weight.detach(), dim=-1).T  # B x Z
        else:
            activations = F.normalize(visual_features, dim=-1) @ F.normalize(self.concepts.weight.detach(), dim=-1).T  # B x Z

        # Highlight (rescale) concept activations with a per-sample saliency score
        scores = self.visual_highlighter_proj(visual_features)
        highlighted_activations = activations * scores

        # Reconstruct visual features from the highlighted activations
        rec_weights, rec_visual_features = self._reconstruct(highlighted_activations)

        if return_weights:
            return rec_visual_features, rec_weights

        return rec_visual_features

    def forward(self, visual_features: torch.Tensor, tags: torch.Tensor | None = None, **kwargs) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass through the interpretable model.

        Reconstructs the visual features, then computes the total training loss as a
        weighted sum of a visual reconstruction loss, a concepts-based disentanglement
        loss, and a regularization loss.

        Parameters
        ----------
        visual_features : torch.Tensor
            Visual features
        tags : Optional[torch.Tensor], optional
            Optional sample-level tags used to determine positive/negative pairs for
            the disentanglement loss, by default None (every sample is treated as
            having a unique, unmatched tag)
        **kwargs
            Additional keyword arguments (e.g. dataset metadata), ignored.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
            Reconstructed visual features, total loss, and a dict of the individual
            (unweighted) loss components.
        """
        if tags is None:
            tags = torch.zeros((visual_features.shape[0], 1), device=self.device)

        rec_visual_features, rec_weights = self.encode_image(visual_features, return_weights=True)

        # Visual reconstruction loss: ensures intra-modal similarities are respected
        visual_rec_loss = self.clip_loss(rec_visual_features, visual_features.detach(), self.visual_rec_temperature)
        if self.gram_regularization:
            # Gram matrix regularization to better preserve intra-modal similarities
            gram_rec = F.normalize(rec_visual_features, dim=-1) @ F.normalize(rec_visual_features, dim=-1).T
            gram_orig = F.normalize(visual_features.detach(), dim=-1) @ F.normalize(visual_features.detach(), dim=-1).T
            visual_rec_loss += F.mse_loss(gram_rec, gram_orig)

        # Concepts-based disentanglement loss: pulls the reconstruction weights of
        # positively-tagged samples together (and, if symmetric, pushes negatives apart)
        concepts_loss = self.concepts_loss(rec_weights, tags, symmetric=self.symm_jsd)

        # Regularization loss on the reconstruction weights (normalized so it is
        # bounded between 0 and 1 for the "entropy" strategy)
        if self.reg_strategy == "dirichlet":
            reg_loss = -((self.dirichlet_alpha - 1) * torch.log(rec_weights + 1e-8)).mean(dim=-1).mean()
        elif self.reg_strategy == "entropy":
            reg_loss = -((rec_weights * (rec_weights + 1e-8).log()).sum(dim=-1) / torch.log(torch.tensor(rec_weights.shape[-1], device=self.device))).mean()
        elif self.reg_strategy == "l1":
            reg_loss = rec_weights.abs().mean()

        # Total loss: weighted sum of the individual components
        loss = self.loss_weight_concepts * concepts_loss + self.loss_weight_rec * visual_rec_loss + self.loss_weight_ent_reg * reg_loss

        return (
            rec_visual_features,
            loss,
            {
                "loss": loss,
                "concepts_loss": concepts_loss,
                "visual_rec_loss": visual_rec_loss,
                "reg_loss": reg_loss,
            },
        )

    def concepts_loss(self, weights: torch.Tensor, tags: torch.Tensor, symmetric: bool = False, epsilon: float = 1e-8) -> torch.Tensor:
        """Compute the concepts disentanglement loss.

        Pulls the reconstruction weight distributions of samples that share at least
        one tag closer together (in Jensen-Shannon divergence), so that samples
        depicting the same underlying concepts get similar reconstructions. If
        `symmetric`, it additionally pushes apart the distributions of samples that
        share no tag.

        Parameters
        ----------
        weights : torch.Tensor
            Reconstruction weights (distributions over concepts), shape (B, num_concepts).
        tags : torch.Tensor
            Sample-level tags, shape (B, num_tags). Two samples are considered a
            "positive" pair if their tag vectors have a positive dot product.
        symmetric : bool, optional
            Whether to compute a symmetric loss, by default False
        epsilon : float, optional
            Small value to avoid division by zero, by default 1e-8

        Returns
        -------
        torch.Tensor
            Concepts disentanglement loss.
        """
        weights_log = (weights + 1e-8).log()

        tags = tags.to(device=self.device).float()

        # Positive pairs: samples sharing at least one tag (upper triangle, no self-pairs)
        positive_targets = (tags @ tags.T > 0).float().fill_diagonal_(0)
        positive_targets = torch.triu(positive_targets, diagonal=1)

        # Negative pairs: samples sharing no tag (upper triangle, no self-pairs)
        negative_targets = (tags @ tags.T == 0).float().fill_diagonal_(0)
        negative_targets = torch.triu(negative_targets, diagonal=1)

        # Pairwise Jensen-Shannon divergence between all samples' reconstruction weights
        if weights.shape[0] ** 2 * weights.shape[1] > _MAX_PAIRWISE_ELEMENTS:
            # Too large to materialize (e.g. ImageNet: 512^2 x 15k concepts, ~16 GB per intermediate):
            # same values, computed a few rows at a time by a compiled kernel and recomputed in backward.
            jsd = _chunked_pairwise_jsd(weights)
        else:
            mixture = (0.5 * (weights[None, :] + weights[:, None]) + 1e-8).log()
            jsd = 0.5 * (
                # Entropy term                               Cross entropy term
                ((weights[None, :] * weights_log[None, :]) - weights[None, :] * mixture).sum(-1) + ((weights[:, None] * weights_log[:, None]) - weights[:, None] * mixture).sum(-1)
            )

        if symmetric:
            return (jsd * positive_targets).sum() / (positive_targets.sum() + epsilon) - (jsd * negative_targets).sum() / (negative_targets.sum() + epsilon)

        return (jsd * positive_targets).sum() / (positive_targets.sum() + epsilon)

    @property
    def device(self) -> torch.device:
        """Get the device where the model is located.

        Returns
        -------
        torch.device
            device where the model is located
        """
        return next(self.parameters()).device

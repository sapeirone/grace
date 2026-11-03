"""
The visual mean: the mean of the L2-normalized visual embeddings of a dataset, renormalized. SpLiCE
always centers on it; GRACE only when `center_before_activations` is set (the image checkpoints).

Which split each model uses:
- SpLiCE on video (EgoMCQ, Charades-Ego): EgoClip train, cached in `data/means/egoclip_<backbone>.pt`.
- SpLiCE on images: the evaluated split itself, cached in `data/means/<dataset>_<split>_<backbone>.pt`.
- GRACE on images: ImageNet train, computed by `train_image.py` and saved in the checkpoint.
- GRACE on video: none (no centering).
"""

import logging
import os
from collections.abc import Callable, Iterable

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@torch.inference_mode()
def compute_visual_mean(embeddings: Iterable[torch.Tensor]) -> torch.Tensor:
    """Mean of the L2-normalized embeddings (an iterable of `(N, D)` batches), renormalized, on the CPU.
    Accumulated in float64."""
    total, count = None, 0
    for batch in tqdm(embeddings, desc="Computing the visual mean...", leave=False):
        batch_sum = F.normalize(batch.float(), p=2, dim=-1).double().sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        count += batch.shape[0]
    return F.normalize((total / count).float(), p=2, dim=-1).cpu()


def load_or_compute_visual_mean(path: str | None, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
    """The visual mean cached at `path`, or `compute()`d and saved there (not saved if `path` is None)."""
    if path is not None and os.path.exists(path):
        logger.info("Loading the visual mean from %s...", path)
        return torch.load(path, map_location="cpu", weights_only=True)
    logger.info("Computing the visual mean...")
    mean = compute()
    if path is not None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(mean, path)
        logger.info("Saved the visual mean to %s", path)
    return mean

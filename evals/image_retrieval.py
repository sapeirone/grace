"""
Entrypoint for evaluating a Grace image checkpoint (or a training-free model, e.g. SpLiCE) via image-text retrieval
(COCO / Flickr30k).

Retrieval is computed in-batch: each batch's image/text embeddings form a square
similarity matrix, R@k is computed against its diagonal, and metrics are averaged across
batches. `cfg.batch_size` therefore directly determines the reported numbers, not just
throughput — see the `batch_size: 1000` override in `configs/experiments/grace/{coco,flickr30k}.yaml`.
"""

import logging

import hydra
import omegaconf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.transforms import get_clip_val_caption_transform
from evals.image_classification import build_model
from evals.metrics import RunningRetrievalStats
from models.backbones.base import CLIP
from models.base import InterpretableCLIP
from utils.entrypoint import log_banner

logger = logging.getLogger(__name__)


@torch.inference_mode()
def eval_retrieval(backbone: CLIP, model: InterpretableCLIP, loader: DataLoader, device: torch.device) -> RunningRetrievalStats:
    """Run image-text retrieval and report performance/faithfulness R@1/5/10.

    Parameters
    ----------
    backbone : CLIP
        The (frozen) CLIP backbone used to encode images and captions.
    model : InterpretableCLIP
        Concept-based model (Grace checkpoint or SpLiCE).
    loader : DataLoader
        Dataloader over (image, tokenized first caption) pairs.
    device : torch.device
        Device to run the evaluation on.

    Returns
    -------
    RunningRetrievalStats
        Accumulated per-batch retrieval metrics; call `.summary()` for the final report.
    """
    model.eval()
    backbone.eval()

    stats = RunningRetrievalStats()

    pbar = tqdm(loader, desc="Evaluating retrieval...", leave=False)
    for images, captions in pbar:
        images, captions = images.to(device), captions.to(device)

        visual_features = backbone.encode_image(images)
        text_features = backbone.encode_text(captions)

        # Unnormalized input, as in the original retrieval pipeline (classification normalizes it first).
        rec_visual_features = model.encode_image(visual_features)

        target_sims = F.normalize(visual_features, p=2, dim=1) @ F.normalize(text_features, p=2, dim=1).T
        surrogate_sims = F.normalize(rec_visual_features, p=2, dim=1) @ F.normalize(text_features, p=2, dim=1).T

        stats.update(surrogate_sims, target_sims)

    logger.info("Retrieval summary:\n%s", stats.summary().round(2).to_string())
    return stats


@hydra.main(config_path="../configs/experiments/", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_banner(logger)

    backbone = hydra.utils.instantiate(cfg.backbone).to(device)

    logger.info("Preparing the dataset...")
    dataset = hydra.utils.instantiate(
        cfg.dataset,
        transform=backbone.preprocess,
        target_transform=get_clip_val_caption_transform(),
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    logger.info("Preparing the dataset... Done! (%d samples)", len(dataset))

    logger.info("Preparing the model...")
    model = build_model(cfg, backbone, loader, device)
    logger.info("Preparing the model... Done!")

    eval_retrieval(backbone, model, loader, device)

    logger.info("All done!")


if __name__ == "__main__":
    main()

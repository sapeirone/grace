"""
Entrypoint for evaluating a video model zero-shot on Charades-Ego action recognition.
"""

import logging

import hydra
import omegaconf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.aggregation import MeanMetric
from tqdm.auto import tqdm

from evals.egomcq import build_model
from evals.metrics import charades_map
from models.backbones.base import CLIP
from models.base import InterpretableCLIP
from utils.entrypoint import log_banner

logger = logging.getLogger(__name__)


@torch.no_grad()
def eval_charades(
    backbone: CLIP,
    model: InterpretableCLIP,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Eval a video model on Charades-Ego zero-shot action recognition.

    Each video is scored against the text embeddings of the 157 action labels (cosine similarity),
    and the scores are evaluated with the multi-label mAP.

    Parameters
    ----------
    backbone : CLIP
        frozen backbone, used to encode the videos and the action labels
    model : InterpretableCLIP
        concept-based model, applied on top of the backbone's video embeddings
    loader : DataLoader
        Dataloader for the Charades-Ego dataset (yielding `(video_id, frames, labels)`)
    device : torch.device
        Device to run the evaluation on.

    Returns
    -------
    dict[str, float]
        `map`: mAP of the concept-based (reconstructed) embedding; `faithfulness`: how often its top-1
        action is the same as the backbone embedding's; `baseline_map`: mAP of the backbone embedding
        (no concepts bottleneck) (all percentages); `l0_norm`: mean number of non-zero concept weights.
    """
    model.eval()

    labels_embds = F.normalize(backbone.encode_text(loader.dataset.labels()).to(device), p=2, dim=-1)

    l0_norm = MeanMetric().to(device=device)
    faithfulness = MeanMetric().to(device=device)

    sims, baseline_sims, labels = [], [], []
    for _, frames, batch_labels in tqdm(loader, desc="Validating on Charades-Ego...", leave=False):
        video_features = backbone.encode_image(frames.to(device))
        batch_baseline_sims = F.normalize(video_features, p=2, dim=-1) @ labels_embds.T

        surrogate_video_features, weights = model.encode_image(video_features, return_weights=True)
        l0_norm.update(torch.linalg.vector_norm(weights, ord=0, dim=-1).float())

        batch_sims = F.normalize(surrogate_video_features, p=2, dim=-1) @ labels_embds.T
        faithfulness.update((batch_baseline_sims.argmax(-1) == batch_sims.argmax(-1)).float())

        sims.append(batch_sims.cpu())
        baseline_sims.append(batch_baseline_sims.cpu())
        labels.append(batch_labels.cpu())

    labels = torch.vstack(labels).numpy()
    m_ap, *_ = charades_map(torch.vstack(sims).numpy(), labels)
    baseline_m_ap, *_ = charades_map(torch.vstack(baseline_sims).numpy(), labels)

    logger.info(
        "Charades-Ego mAP = %.2f (baseline %.2f). L0-Norm = %.3f. Faithfulness = %.2f",
        100 * m_ap,
        100 * baseline_m_ap,
        l0_norm.compute(),
        100 * faithfulness.compute(),
    )
    return {
        "map": float(100 * m_ap),
        "faithfulness": float(100 * faithfulness.compute()),
        "baseline_map": float(100 * baseline_m_ap),
        "l0_norm": float(l0_norm.compute()),
    }


@hydra.main(config_path="../configs/experiments/", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_banner(logger)

    # Step 1: build the model
    logger.info("")
    logger.info("Preparing the model...")
    backbone = hydra.utils.instantiate(cfg.backbone).to(device)
    model = build_model(cfg, backbone, device)
    logger.info("Preparing the model... Done!")

    # Step 2: dataset preparation (raw videos, decoded on the fly)
    logger.info("")
    logger.info("Loading the Charades-Ego dataset...")
    val_dataset = hydra.utils.instantiate(cfg.val_dataset, transforms=backbone.preprocess)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    logger.info("# val batches: %d", len(val_loader))

    eval_charades(backbone, model, val_loader, device)

    logger.info("All done!")


if __name__ == "__main__":
    main()

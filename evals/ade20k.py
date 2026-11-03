"""
Entrypoint for evaluating a GRACE image checkpoint on ADE20k concept localization.

Every patch embedding of the backbone (TIPSv2: a 32x32 grid at 448px) is decomposed by GRACE into
reconstruction weights over the concept vocabulary. A concept is a good detector for an ADE20k
class if ranking the image's patches by that concept's weight puts the class's patches first, as
measured by the average precision (AUPRC) of that ranking against the patch labels. On the
training split, each of the 150 classes is matched to the concept with the highest AP averaged
over the images containing the class; the score is that concept's AP on the validation split,
averaged per class and then over the 150 classes (the ConceptScope protocol, Table 9 of the paper).
"""

import json
import logging
import os

import hydra
import omegaconf
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.backbones.tipsv2 import TIPSv2Backbone
from models.checkpoint import load_checkpoint
from models.grace import Grace
from utils.entrypoint import log_banner

logger = logging.getLogger(__name__)


@torch.no_grad()
def patch_weights(backbone: TIPSv2Backbone, model: Grace, image: torch.Tensor) -> torch.Tensor:
    """GRACE reconstruction weights of every patch of one image, `(num_patches, num_concepts)`.

    Patch embeddings are passed to GRACE unnormalized (GRACE normalizes them before the concept
    match; the highlighter sees them as they are).
    """
    _, patches = backbone.encode_image(image[None], return_patches=True)
    _, weights = model.encode_image(patches.reshape(-1, patches.shape[-1]), return_weights=True)
    return weights


@torch.no_grad()
def class_aps(scores: torch.Tensor, gt: torch.Tensor, num_classes: int) -> dict[int, torch.Tensor]:
    """Average precision of ranking one image's patches by each concept's score, for each class
    present in the image.

    The patches are sorted once, over all concepts together: ties between patches are frequent,
    since most concepts get a ~0 weight, so redoing the sort per class or per concept would change
    the order of tied patches, and the APs.

    Parameters
    ----------
    scores : torch.Tensor
        `(num_patches, num_concepts)` scores
    gt : torch.Tensor
        `(num_patches,)` patch labels (0 = other, `ci + 1` = class `ci`)
    num_classes : int
        number of classes

    Returns
    -------
    dict[int, torch.Tensor]
        for each class `ci` present in the image, the `(num_concepts,)` AP of each concept
    """
    order = torch.argsort(scores, dim=0, descending=True)
    ranks = torch.arange(1, scores.shape[0] + 1, device=scores.device, dtype=torch.float32).view(-1, 1)
    aps = {}
    for ci in range(num_classes):
        y = (gt == ci + 1).float()
        n_pos = y.sum()
        if n_pos == 0:
            continue  # class absent from this image
        y_sorted = y[order]
        precision = torch.cumsum(y_sorted, dim=0) / ranks
        aps[ci] = (precision * y_sorted).sum(dim=0) / n_pos
    return aps


@torch.no_grad()
def eval_ade20k(
    backbone: TIPSv2Backbone,
    model: Grace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict]:
    """Match each ADE20k class to a concept on `train_loader`, then score it on `val_loader`.

    Returns
    -------
    tuple[float, dict]
        the mean (over the 150 classes) validation AUPRC of each class's matched concept, and the
        per-class details (`class_name`, `latent_idx`, `concept`, train `auprc`, `val_auprc`)
    """
    model.eval()
    classes = train_loader.dataset.labels()
    num_concepts = len(model.text_concepts)

    # Step 1: per-class AP of every concept, averaged over the training images containing the class.
    train_ap_sum = torch.zeros(len(classes), num_concepts, device=device)
    train_class_counts = torch.zeros(len(classes), device=device)
    for images, labels, _ in tqdm(train_loader, desc="Matching classes to concepts (ADE20k training)...", leave=False):
        for image, image_labels in zip(images, labels, strict=True):
            weights = patch_weights(backbone, model, image.to(device))
            for ci, ap in class_aps(weights, image_labels.reshape(-1).to(device), len(classes)).items():
                train_ap_sum[ci] += ap
                train_class_counts[ci] += 1
    best_ap_train, best_concept_train = (train_ap_sum / train_class_counts.clamp(min=1).unsqueeze(1)).max(dim=1)

    # Step 2: AP of each class's matched concept on the validation images containing the class.
    val_aps = [[] for _ in classes]
    for images, labels, _ in tqdm(val_loader, desc="Scoring the matched concepts (ADE20k validation)...", leave=False):
        for image, image_labels in zip(images, labels, strict=True):
            weights = patch_weights(backbone, model, image.to(device))
            for ci, ap in class_aps(weights, image_labels.reshape(-1).to(device), len(classes)).items():
                val_aps[ci].append(ap[best_concept_train[ci]].item())

    # Classes never present in validation count as 0.
    val_ap_by_class = [sum(aps) / len(aps) if aps else 0.0 for aps in val_aps]
    mean_auprc = sum(val_ap_by_class) / len(classes)

    details = {
        ci: {
            "class_name": classes[ci],
            "latent_idx": int(best_concept_train[ci]),
            "concept": model.text_concepts[int(best_concept_train[ci])],
            "auprc": float(best_ap_train[ci]),
            "val_auprc": val_ap_by_class[ci],
        }
        for ci in range(len(classes))
    }
    logger.info("ADE20k concept localization: mean AUPRC = %.4f (train-set matching AUPRC %.4f)", mean_auprc, best_ap_train.mean().item())
    return mean_auprc, details


@hydra.main(config_path="../configs/experiments/", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_banner(logger)

    backbone = hydra.utils.instantiate(cfg.backbone).eval().to(device)
    model = load_checkpoint(cfg.resume_from, device)

    loaders = {
        split: DataLoader(
            hydra.utils.instantiate(cfg.dataset, split=split, transforms=backbone.preprocess),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        for split in ("training", "validation")
    }

    _, details = eval_ade20k(backbone, model, loaders["training"], loaders["validation"], device)

    # Save the class -> concept matching next to the run's logs.
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    with open(os.path.join(output_dir, "latent_class_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(details, f, indent=4)
    logger.info("Saved the class-to-concept matching to %s", os.path.join(output_dir, "latent_class_mapping.json"))

    logger.info("All done!")


if __name__ == "__main__":
    main()

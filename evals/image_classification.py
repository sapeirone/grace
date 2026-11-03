"""
Entrypoint for evaluating a trained Grace image checkpoint via zero-shot classification
(ImageNet / Flowers102 / Food101).
"""

import json
import logging

import hydra
import omegaconf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics.aggregation import MeanMetric
from tqdm.auto import tqdm

from data.imagenet import ImageNetDataset
from evals.metrics import find_closest, get_classes
from models.backbones.base import CLIP
from models.base import InterpretableCLIP
from models.checkpoint import load_checkpoint
from utils.concepts import load_concepts
from utils.entrypoint import log_banner
from utils.visual_mean import compute_visual_mean, load_or_compute_visual_mean

logger = logging.getLogger(__name__)


@torch.inference_mode()
def eval_classification(
    backbone: CLIP,
    model: InterpretableCLIP,
    loader: DataLoader,
    label_embeddings: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run zero-shot classification and report accuracy and faithfulness.

    Parameters
    ----------
    backbone : CLIP
        The (frozen) CLIP backbone used to encode images and the class-name prompts.
    model : InterpretableCLIP
        Concept-based model (Grace checkpoint or SpLiCE).
    loader : DataLoader
        Dataloader over (image, label) pairs.
    label_embeddings : torch.Tensor
        L2-normalized zero-shot label embeddings ("a photo of a {class}."), shape (num_classes, D).
    device : torch.device
        Device to run the evaluation on.

    Returns
    -------
    tuple[float, float, float]
        Accuracy, faithfulness and baseline (raw CLIP) accuracy, as percentages.
    """
    model.eval()
    backbone.eval()

    accuracy = MeanMetric().to(device=device)
    # Baseline accuracy of the raw CLIP embedding (no concepts bottleneck), for comparison.
    baseline_accuracy = MeanMetric().to(device=device)
    # Faithfulness: how often the Grace reconstruction agrees with the raw CLIP prediction.
    faithfulness = MeanMetric().to(device=device)

    pbar = tqdm(loader, desc="Evaluating zero-shot classification...", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        original_embedding = F.normalize(backbone.encode_image(images), dim=-1)
        target_preds = find_closest(original_embedding, label_embeddings)

        # The model gets the L2-normalized embedding here, as in the original classification pipeline
        # (retrieval, Charades and ADE20k pass it unnormalized); the released numbers depend on it.
        rec_embedding = model.encode_image(original_embedding)
        rec_embedding = F.normalize(rec_embedding, dim=-1)
        preds = find_closest(rec_embedding, label_embeddings)

        accuracy.update((preds == labels).float())
        baseline_accuracy.update((target_preds == labels).float())
        faithfulness.update((preds == target_preds).float())

        pbar.set_description(f"Accuracy = {100 * accuracy.compute():.2f}, Faithfulness = {100 * faithfulness.compute():.2f}")

    logger.info(
        "Accuracy = %.2f. Faithfulness = %.2f. Baseline (CLIP) accuracy = %.2f",
        100 * accuracy.compute(),
        100 * faithfulness.compute(),
        100 * baseline_accuracy.compute(),
    )
    return float(100 * accuracy.compute()), float(100 * faithfulness.compute()), float(100 * baseline_accuracy.compute())


@hydra.main(config_path="../configs/experiments/", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_banner(logger)

    backbone = hydra.utils.instantiate(cfg.backbone).to(device)

    logger.info("Preparing the dataset...")
    dataset = hydra.utils.instantiate(cfg.dataset, transform=backbone.preprocess)
    if cfg.get("folds_path") is not None:
        dataset = restrict_to_fold_test_classes(dataset, cfg.folds_path, cfg.fold_id)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    logger.info("Preparing the dataset... Done! (%d samples)", len(dataset))

    logger.info("Preparing the model...")
    model = build_model(cfg, backbone, loader, device)
    logger.info("Preparing the model... Done!")

    # With a fold, images are restricted to its test classes but still classified among all classes.
    classes = get_classes(cfg.dataset_name, dataset=dataset)
    logger.info("Number of classes: %d", len(classes))
    label_embeddings = zero_shot_label_embeddings(backbone, classes)

    eval_classification(backbone, model, loader, label_embeddings, device)

    logger.info("All done!")


@torch.inference_mode()
def zero_shot_label_embeddings(backbone: CLIP, classes: list[str]) -> torch.Tensor:
    """L2-normalized text embeddings of the zero-shot prompts "a photo of a {class}.", `(num_classes, D)`."""
    return F.normalize(backbone.encode_text([f"a photo of a {name}." for name in classes]), dim=-1)


def restrict_to_fold_test_classes(dataset: ImageNetDataset, folds_path: str, fold_id: int) -> Subset:
    """Keep only the samples whose label is in fold `fold_id`'s `test_classes` (ImageNet only)."""
    with open(folds_path, encoding="utf-8") as f:
        test_classes = set(json.load(f)["folds"][fold_id]["test_classes"])
    keep = [i for i, target in enumerate(dataset.targets) if target in test_classes]
    logger.info("Fold %d: %d test classes, %d / %d samples (from %s)", fold_id, len(test_classes), len(keep), len(dataset), folds_path)
    return Subset(dataset, keep)


def build_model(cfg: omegaconf.DictConfig, backbone: CLIP, loader: DataLoader, device: torch.device) -> InterpretableCLIP:
    """Build the image model to evaluate: the checkpoint at `cfg.resume_from` if set, otherwise a model
    that needs no checkpoint (e.g. SpLiCE) from `cfg.model`, the concept vocabulary at `cfg.concepts_path`
    and the visual mean of the evaluated split, cached at `cfg.visual_mean_path` (computed over `loader`
    if missing). A null `visual_mean_path` gives a zero mean, i.e. no centering (for ablations)."""
    if cfg.resume_from is not None:
        return load_checkpoint(cfg.resume_from, device)

    with torch.inference_mode():
        text_concepts, concepts_embds = load_concepts(cfg.concepts_path, backbone)
    model = hydra.utils.instantiate(cfg.model, features_size=backbone.features_size, text_concepts=text_concepts, concepts_embds=concepts_embds.clone())
    if cfg.visual_mean_path is None:
        visual_mean = torch.zeros(backbone.features_size)
    else:
        visual_mean = load_or_compute_visual_mean(cfg.visual_mean_path, lambda: compute_visual_mean(backbone.encode_image(images.to(device)) for images, *_ in loader))
    model.set_visual_mean(visual_mean)
    return model.to(device).eval()


if __name__ == "__main__":
    main()

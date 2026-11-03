"""
Entrypoint for evaluating a trained model checkpoint on the EgoMCQ benchmark.
"""

import logging

import hydra
import omegaconf
import torch
from einops import rearrange
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.aggregation import MeanMetric
from tqdm.auto import tqdm

from models.backbones.base import CLIP
from models.base import InterpretableCLIP
from models.checkpoint import load_checkpoint
from utils.concepts import load_concepts
from utils.entrypoint import log_banner
from utils.visual_mean import compute_visual_mean, load_or_compute_visual_mean

logger = logging.getLogger(__name__)


@torch.no_grad()
def eval_mcq(
    model: InterpretableCLIP,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Eval the CLIP CBM model on the EgoMCQ dataset.

    Parameters
    ----------
    model : InterpretableCLIP
        CLIP-like CBM model
    loader : DataLoader
        Dataloader for the EgoMCQ dataset.
    device : torch.device
        Device to run the evaluation on.

    Returns
    -------
    dict[str, float]
        `intra` / `inter`: EgoMCQ accuracy of the concept-based surrogate embedding;
        `baseline_intra` / `baseline_inter`: accuracy of the underlying (baseline) CLIP embedding;
        `faithfulness`: how often the two pick the same answer (all percentages);
        `l0_norm`: mean number of non-zero concept weights; `top25_mass`: mean weight of the top-25 concepts.
    """
    model.eval()

    # EgoMCQ accuracy, split by question type (see the type check below).
    inter_meter, intra_meter = MeanMetric().to(device=device), MeanMetric().to(device=device)
    # Baseline accuracy of the underlying CLIP embedding (no concepts bottleneck), for comparison.
    baseline_inter_meter, baseline_intra_meter = MeanMetric().to(device=device), MeanMetric().to(device=device)

    # Concept-level diagnostics for the surrogate (concept-based) embedding:
    # - l0_norm: how many concepts get a non-zero weight per sample (sparsity of the explanation).
    # - top25_mass: how much of the total concept weight is captured by just the top-25 concepts.
    # - faithfulness: how often the surrogate embedding yields the same MCQ answer as the real one.
    l0_norm = MeanMetric().to(device=device)
    top25_mass = MeanMetric().to(device=device)
    faithfulness = MeanMetric().to(device=device)

    pbar = tqdm(loader, desc="Validating on EgoMCQ...", leave=False)
    for visual_features, text_features, _, metadata in pbar:
        visual_features, text_features = visual_features.to(device), text_features.to(device)

        # Each sample is a query with `num_choices` candidate answers; flatten them into the
        # batch dimension so the whole batch can be encoded in a single forward pass.
        batch_size, num_choices, *_ = visual_features.shape
        visual_features = rearrange(visual_features, "B N C -> (B N) C")
        surrogate_visual_features, weights = model.encode_image(visual_features, return_weights=True)

        top25_mass.update(weights.topk(25, dim=-1).values.sum(1).mean().item())
        l0_norm.update(torch.linalg.vector_norm(weights, ord=0, dim=-1).float())

        # Restore the (batch, choice) structure now that per-sample concept weights are logged.
        visual_features = rearrange(visual_features, "(B N) C -> B N C", B=batch_size, N=num_choices)
        surrogate_visual_features = rearrange(surrogate_visual_features, "(B N) C -> B N C", B=batch_size, N=num_choices)

        visual_features = F.normalize(visual_features, p=2, dim=-1)
        surrogate_visual_features = F.normalize(surrogate_visual_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # For each query, pick the candidate whose embedding is closest to the query text.
        preds = torch.bmm(visual_features, text_features[..., None]).squeeze(-1).argmax(-1)
        surrogate_preds = torch.bmm(surrogate_visual_features, text_features[..., None]).squeeze(-1).argmax(-1)

        # Faithfulness: does the interpretable surrogate agree with the underlying CLIP model?
        faithfulness.update((preds == surrogate_preds).float())

        # EgoMCQ has two question types: "inter-video" (type 1) and "intra-video" (type 2).
        for sample_pred, baseline_pred, sample_type, sample_answer in zip(surrogate_preds, preds, metadata["type"], metadata["answer"], strict=True):
            if int(sample_type) == 1:  # inter
                inter_meter.update(sample_pred == sample_answer.item())  # type: ignore
                baseline_inter_meter.update(baseline_pred == sample_answer.item())  # type: ignore
            else:  # intra
                intra_meter.update(sample_pred == sample_answer.item())  # type: ignore
                baseline_intra_meter.update(baseline_pred == sample_answer.item())  # type: ignore

        pbar.set_description(
            f"EgoMCQ Accuracy: intra = {100 * intra_meter.compute():.3f} (baseline {100 * baseline_intra_meter.compute():.3f}), "
            f"inter = {100 * inter_meter.compute():.3f} (baseline {100 * baseline_inter_meter.compute():.3f}). "
            f"L0-Norm = {l0_norm.compute():.3f}. Faithfulness = {100 * faithfulness.compute():.2f}"
        )

    logger.info(
        "EgoMCQ Accuracy: intra = %.3f (baseline %.3f), inter = %.3f (baseline %.3f). L0-Norm = %.3f. Faithfulness = %.2f. Top25 Mass = %.3f",
        100 * intra_meter.compute(),
        100 * baseline_intra_meter.compute(),
        100 * inter_meter.compute(),
        100 * baseline_inter_meter.compute(),
        l0_norm.compute(),
        100 * faithfulness.compute(),
        top25_mass.compute(),
    )
    return {
        "intra": float(100 * intra_meter.compute()),
        "inter": float(100 * inter_meter.compute()),
        "baseline_intra": float(100 * baseline_intra_meter.compute()),
        "baseline_inter": float(100 * baseline_inter_meter.compute()),
        "faithfulness": float(100 * faithfulness.compute()),
        "l0_norm": float(l0_norm.compute()),
        "top25_mass": float(top25_mass.compute()),
    }


def build_model(cfg: omegaconf.DictConfig, backbone: CLIP, device: torch.device) -> InterpretableCLIP:
    """Build the video model to evaluate: the checkpoint at `cfg.resume_from` if set, otherwise a
    training-free model (e.g. SpLiCE) from `cfg.model`, the vocabulary at `cfg.concepts_path` and the
    EgoClip visual mean (cached at `cfg.visual_mean_path`, computed on the first run).

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        hydra config
    backbone : CLIP
        frozen backbone, used to encode the concepts vocabulary when not resuming
    device : torch.device
        device to build the model on

    Returns
    -------
    InterpretableCLIP
        the model, in eval mode
    """
    if cfg.resume_from is not None:
        return load_checkpoint(cfg.resume_from, device)

    text_concepts, concepts_embds = load_concepts(cfg.concepts_path, backbone)
    model = hydra.utils.instantiate(cfg.model, features_size=backbone.features_size, text_concepts=text_concepts, concepts_embds=concepts_embds)

    def compute() -> torch.Tensor:
        logger.info("Loading the EgoClip dataset...")
        train_dataset = hydra.utils.instantiate(cfg.train_dataset, transforms=backbone.preprocess, ret_videos=False)
        # In-order reads: shuffled access to the mmapped features file is ~100x slower.
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
        return compute_visual_mean(features for features, *_ in train_loader)

    model.set_visual_mean(load_or_compute_visual_mean(cfg.get("visual_mean_path"), compute))
    return model.eval().to(device)


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

    # Step 2: dataset preparation
    logger.info("")
    logger.info("Preparing the datasets...")

    logger.info("Loading the EgoMCQ dataset...")
    val_dataset = hydra.utils.instantiate(cfg.val_dataset, transforms=backbone.preprocess, ret_videos=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    logger.info("# val batches: %d", len(val_loader))

    logger.info("Preparing the datasets... Done!")

    eval_mcq(model, val_loader, device=device)

    logger.info("All done!")


if __name__ == "__main__":
    main()

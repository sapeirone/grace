"""
Entrypoint for training GRACE (video, on precomputed EgoClip features).

`python -m train --config-name grace/egomcq` trains the paper's GRACE (`checkpoints/video/grace.pth`'s
recipe, the defaults of `configs/components/model/grace.yaml`); the checkpoint is saved under
`checkpoints/runs/` and can be evaluated with `evals/egomcq.py resume_from=...`.
"""

import logging
import os
from datetime import datetime

import hydra
import omegaconf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.aggregation import MeanMetric
from tqdm.auto import tqdm

from evals.egomcq import eval_mcq
from models.base import InterpretableCLIP
from models.checkpoint import checkpoint_dict
from models.grace import Grace
from utils.concepts import load_concepts
from utils.entrypoint import log_banner
from utils.training import LossMeter, seed_everything, to_dict
from utils.visual_mean import compute_visual_mean

logger = logging.getLogger(__name__)


def train_epoch(
    model: InterpretableCLIP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,  # type: ignore
    train_loader: torch.utils.data.DataLoader,
    meter: LossMeter,
    epoch: int,
    device: torch.device,
) -> None:
    """Train the model for one epoch.

    Parameters
    ----------
    model : InterpretableCLIP
        The interpretable CLIP model to train
    optimizer : torch.optim.Optimizer
        The optimizer to use for training
    scheduler : torch.optim.lr_scheduler.LRScheduler
        The lr scheduler to use for training (stepped every iteration)
    train_loader : torch.utils.data.DataLoader
        The training dataloader
    meter : LossMeter
        The loss meter (running means over the epoch)
    epoch : int
        The current epoch number
    device : torch.device
        The device to use for training.
    """
    model.train()
    meter.reset()

    for visual_features, *_, metadata in (pbar := tqdm(train_loader, total=len(train_loader), leave=False)):
        visual_features = visual_features.to(device)

        # Forward through the model (the losses only use the visual features and the metadata's tags)
        _, loss, losses = model(visual_features, **metadata)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        # Update the meter and the progress bar
        meter.update(**losses)
        pbar.set_description(f"Epoch {epoch + 1}: {meter.log()}, lr={optimizer.param_groups[0]['lr']:.2e}")

    logger.info("Epoch n. %d completed! %s", epoch + 1, meter.log())


@torch.no_grad()
def eval_model(
    model: InterpretableCLIP,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate the losses and the in-batch text retrieval faithfulness on a dataset.

    Parameters
    ----------
    model : InterpretableCLIP
        The interpretable CLIP model to evaluate
    loader : torch.utils.data.DataLoader
        The dataloader for evaluation
    device : torch.device
        The device to use.

    Returns
    -------
    tuple[float, float, float]
        Mean loss, reconstruction loss and faithfulness.
    """
    # The losses are computed in train mode, as during training.
    model.train()

    loss_meter = MeanMetric().to(device=device)
    reconstruction_loss_meter = MeanMetric().to(device=device)
    faithfulness_meter = MeanMetric().to(device=device)

    for visual_features, text_features, *_, metadata in tqdm(loader, total=len(loader), leave=False):
        visual_features, text_features = visual_features.to(device), text_features.to(device)

        # Forward through the model
        surrogate_visual_features, _, losses = model(visual_features, **metadata)

        loss_meter.update(losses["loss"])
        reconstruction_loss_meter.update(losses["visual_rec_loss"])

        sims = F.normalize(visual_features, p=2, dim=-1) @ F.normalize(text_features, p=2, dim=-1).T
        surrogate_sims = F.normalize(surrogate_visual_features, p=2, dim=-1) @ F.normalize(text_features, p=2, dim=-1).T
        faithfulness_meter.update(sims.argmax(-1) == surrogate_sims.argmax(-1))

    logger.info(
        "Evaluation on training set: Loss = %.4f, Reconstruction Loss = %.4f, Faithfulness = %.4f",
        loss_meter.compute(),
        reconstruction_loss_meter.compute(),
        faithfulness_meter.compute(),
    )
    return float(loss_meter.compute()), float(reconstruction_loss_meter.compute()), float(faithfulness_meter.compute())


@hydra.main(config_path="configs/experiments/", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:
    """Train GRACE on the precomputed EgoClip features, then evaluate it on EgoMCQ.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra config
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_banner(logger)

    seed_everything(cfg.seed)
    logger.info("Seed: %d", cfg.seed)

    # Step 1: build the model
    logger.info("")
    logger.info("Preparing the model...")
    backbone = hydra.utils.instantiate(cfg.backbone).to(device)

    # Step 1.1: load the textual concepts
    text_concepts, concepts_embds = load_concepts(cfg.concepts_path, backbone)

    # Step 1.2: build the full model
    model: Grace = hydra.utils.instantiate(cfg.model, features_size=backbone.features_size, concepts_embds=concepts_embds, text_concepts=text_concepts).to(device)

    logger.info("Preparing the model... Done!")

    # Step 2: dataset preparation
    logger.info("")
    logger.info("Preparing the datasets...")

    # Load the EgoClip dataset for training
    train_dataset = hydra.utils.instantiate(cfg.train_dataset, transforms=backbone.preprocess, ret_videos=False)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # Load the EgoMCQ dataset for validation
    egomcq_dataset = hydra.utils.instantiate(cfg.val_dataset, transforms=backbone.preprocess, ret_videos=False)
    egomcq_loader = DataLoader(egomcq_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    logger.info("# train batches: %d. # val batches: %d", len(train_loader), len(egomcq_loader))

    logger.info("Preparing the datasets... Done!")

    # Step 3: build the optimizer (1 epoch of linear warmup, then cosine decay to 0)
    logger.info("")
    logger.info("Preparing the optimizer...")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=len(train_loader))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(cfg.num_epochs - 1) * len(train_loader), eta_min=0.0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[len(train_loader)])
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Number of trainable parameters: %.2f / %.2f (%2.2f %%) MParams", num_trainable_params, num_params, num_trainable_params / num_params * 100)
    logger.info("Preparing the optimizer... Done!")

    meter = LossMeter(device=device)

    # Step 4: create the save path for checkpoints
    save_path = os.path.join("checkpoints", "runs", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{cfg.name}")
    logger.info("Saving artifacts to %s...", save_path)
    os.makedirs(save_path, exist_ok=True)

    start_epoch = 0
    if cfg.resume_from is not None:
        logger.info("")
        logger.info("Resuming from %s...", cfg.resume_from)
        checkpoint = torch.load(cfg.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)

    # Step 5: actual training
    logger.info("")
    logger.info("Starting the training process...")

    # Visual mean of EgoClip, only if the model centers on it before the concept match (off in the paper
    # recipe); saved in the checkpoint. In-order reads: shuffled reads of the mmapped features are ~100x slower.
    if model.center_before_activations and cfg.resume_from is None:
        mean_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        model.set_visual_mean(compute_visual_mean(features for features, *_ in mean_loader))

    for epoch in range(start_epoch, cfg.num_epochs):
        train_epoch(model, optimizer, scheduler, train_loader, meter, epoch, device=device)

    if cfg.save_checkpoints:
        logger.info("Saving checkpoints to %s...", save_path)
        torch.save(
            checkpoint_dict(
                model,
                to_dict(cfg.model),
                to_dict(cfg.backbone),
                epoch=max(start_epoch, cfg.num_epochs),  # number of completed epochs
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                config=to_dict(cfg),
            ),
            f"{save_path}/model.pth",
        )

    # Final eval on training set
    logger.info("Evaluating the losses on the training set...")
    eval_model(model, val_loader, device=device)

    # Final eval on EgoMCQ
    eval_mcq(model, egomcq_loader, device=device)

    logger.info("All done!")


if __name__ == "__main__":
    main()

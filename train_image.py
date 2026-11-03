"""
Entrypoint for training GRACE on images (ImageNet, on precomputed CLIP features).

`python -m train_image --config-name grace/imagenet_train` trains the paper's image GRACE
(`checkpoints/image/grace.pt`'s recipe, `configs/experiments/grace/imagenet_train.yaml`); the best
checkpoint is saved under `checkpoints/runs/` and can be evaluated with
`evals/image_classification.py resume_from=...` / `evals/image_retrieval.py resume_from=...`.
"""

import logging
import os
from datetime import datetime

import hydra
import omegaconf
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from data.imagenet_features import load_imagenet_features
from models.checkpoint import checkpoint_dict
from models.grace import Grace
from utils.concepts import load_concepts
from utils.entrypoint import log_banner
from utils.training import LossMeter, seed_everything, to_dict
from utils.visual_mean import compute_visual_mean

logger = logging.getLogger(__name__)


def train_epoch(
    model: Grace,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    meter: LossMeter,
    epoch: int,
    generator: torch.Generator,
) -> None:
    """Train the model for one epoch.

    Parameters
    ----------
    model : Grace
        The GRACE model to train
    optimizer : torch.optim.Optimizer
        The optimizer to use for training
    features : torch.Tensor
        The `(N, D)` precomputed features of the training set, on the training device
    labels : torch.Tensor
        The `(N,)` class labels of the training set, on the training device
    batch_size : int
        The batch size (shuffled batches, the last incomplete one is dropped)
    meter : LossMeter
        The loss meter (running means over the epoch)
    epoch : int
        The current epoch number
    generator : torch.Generator
        The generator of the shuffling, on the training device
    """
    model.train()
    meter.reset()

    num_batches = len(features) // batch_size
    num_classes = int(labels.max()) + 1
    order = torch.randperm(len(features), generator=generator, device=features.device)[: num_batches * batch_size]

    for idx in (pbar := tqdm(order.split(batch_size), leave=False)):
        visual_features = features[idx]
        tags = F.one_hot(labels[idx], num_classes).float()  # same-class samples are positives

        # Forward through the model
        _, loss, losses = model(visual_features, tags=tags)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the meter and the progress bar
        meter.update(**losses)
        pbar.set_description(f"Epoch {epoch + 1}: {meter.log()}")

    logger.info("Epoch n. %d completed! %s", epoch + 1, meter.log())


@torch.no_grad()
def eval_model(model: Grace, features: torch.Tensor, batch_size: int, generator: torch.Generator) -> float:
    """Evaluate the reconstruction contrastive loss (reconstructed vs original features, in-batch) on a
    dataset, in eval mode: the checkpoint-selection criterion of `grace.pt`.

    Parameters
    ----------
    model : Grace
        The GRACE model to evaluate
    features : torch.Tensor
        The `(N, D)` precomputed features to evaluate on, on the evaluation device
    batch_size : int
        The batch size (random batches)
    generator : torch.Generator
        The generator of the batches, on the evaluation device

    Returns
    -------
    float
        Mean reconstruction loss.
    """
    model.eval()

    order = torch.randperm(len(features), generator=generator, device=features.device)
    losses = [model.clip_loss(model.encode_image(features[idx]), features[idx], model.visual_rec_temperature).item() for idx in tqdm(order.split(batch_size), leave=False)]
    return sum(losses) / len(losses)


@hydra.main(config_path="configs/experiments/", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:
    """Train GRACE on the precomputed ImageNet features, keeping the best checkpoint.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra config
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_banner(logger)

    seed_everything(cfg.seed)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    logger.info("Seed: %d", cfg.seed)

    # Step 1: build the model
    logger.info("")
    logger.info("Preparing the model...")
    backbone = hydra.utils.instantiate(cfg.backbone).eval().to(device)

    # Step 1.1: load the textual concepts
    text_concepts, concepts_embds = load_concepts(cfg.concepts_path, backbone)

    # Step 1.2: build the full model
    model = hydra.utils.instantiate(cfg.model, features_size=backbone.features_size, concepts_embds=concepts_embds.cpu(), text_concepts=text_concepts).to(device)
    del backbone  # only needed to encode the vocabulary; the image features are precomputed

    logger.info("Preparing the model... Done!")

    # Step 2: dataset preparation (precomputed ImageNet train features, all on the GPU: 1.28M x 512 floats, ~2.6 GB)
    logger.info("")
    logger.info("Preparing the datasets...")
    features, labels = load_imagenet_features(cfg.features_path, cfg.imagenet_root, split="train")
    features, labels = features.to(device), labels.to(device)
    logger.info("# train batches: %d", len(features) // cfg.batch_size)
    logger.info("Preparing the datasets... Done!")

    # Step 3: build the optimizer (Adam, no schedule)
    logger.info("")
    logger.info("Preparing the optimizer...")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Number of trainable parameters: %.2f / %.2f (%2.2f %%) MParams", num_trainable_params, num_params, num_trainable_params / num_params * 100)
    logger.info("Preparing the optimizer... Done!")

    meter = LossMeter(device=device)

    # Step 4: create the save path for checkpoints
    save_path = os.path.join("checkpoints", "runs", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{cfg.name}")
    logger.info("Saving artifacts to %s...", save_path)
    os.makedirs(save_path, exist_ok=True)

    start_epoch, best_val_loss = 0, float("inf")
    if cfg.resume_from is not None:
        logger.info("")
        logger.info("Resuming from %s...", cfg.resume_from)
        checkpoint = torch.load(cfg.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("val_loss", float("inf"))

    # Step 5: actual training
    logger.info("")
    logger.info("Starting the training process...")

    # Visual mean of ImageNet train (the model centers on it before the concept match), saved in the checkpoint
    if cfg.resume_from is None:
        model.set_visual_mean(compute_visual_mean([features]))

    for epoch in range(start_epoch, cfg.num_epochs):
        train_epoch(model, optimizer, features, labels, cfg.batch_size, meter, epoch, generator)

        # Every `val_freq` epochs, keep the checkpoint with the lowest reconstruction loss on the training set
        if epoch % cfg.val_freq == cfg.val_freq - 1:
            logger.info("Evaluating the reconstruction loss on the training set...")
            val_loss = eval_model(model, features, cfg.val_batch_size, generator)
            logger.info("Reconstruction loss: %.6f (best %.6f)", val_loss, best_val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if cfg.save_checkpoints:
                    logger.info("Saving the best checkpoint (epoch %d) to %s...", epoch + 1, save_path)
                    torch.save(
                        checkpoint_dict(
                            model,
                            to_dict(cfg.model),
                            to_dict(cfg.backbone),
                            epoch=epoch + 1,  # number of completed epochs
                            val_loss=val_loss,
                            optimizer_state_dict=optimizer.state_dict(),
                            config=to_dict(cfg),
                        ),
                        f"{save_path}/model.pt",
                    )

    logger.info("All done!")


if __name__ == "__main__":
    main()

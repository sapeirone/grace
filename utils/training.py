"""
Helpers shared by the training entrypoints (`train.py`, `train_image.py`).
"""

import random

import numpy as np
import omegaconf
import torch
from torchmetrics.aggregation import MeanMetric


class LossMeter:
    """Running means of the (unweighted) loss components returned by the model's forward."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.meters: dict[str, MeanMetric] = {}

    @torch.no_grad()
    def update(self, **losses: torch.Tensor) -> None:
        for name, value in losses.items():
            if name not in self.meters:
                self.meters[name] = MeanMetric().to(self.device)
            self.meters[name].update(value)

    def log(self) -> str:
        return ", ".join(f"{name}={meter.compute():.3f}" for name, meter in self.meters.items())

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_dict(config: omegaconf.DictConfig) -> dict:
    """A resolved plain-dict copy of a Hydra config, to save in a checkpoint."""
    return omegaconf.OmegaConf.to_container(config, resolve=True)

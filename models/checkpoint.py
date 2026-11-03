"""
GRACE checkpoints: one format for the released checkpoints and for the ones saved by `train.py` /
`train_image.py`. A checkpoint is a plain dict, loadable with `torch.load(..., weights_only=True)`:

- `format_version`: 1.
- `model`: the model's constructor kwargs (`_target_` plus every argument except `text_concepts` and
  `concepts_embds`, all explicit, defaults included), instantiated with Hydra.
- `backbone`: the frozen backbone the model was trained on (`_target_`, `variant`, ...), as in
  `configs/components/backbone/`. Informative: the evals build the backbone from their own config.
- `text_concepts`: the concept vocabulary (list of strings).
- `model_state_dict`: the model weights, concept embeddings (`concepts.weight`) included.
- `epoch` (optional): the number of completed training epochs.

Checkpoints saved by `train.py` also carry `optimizer_state_dict`, `scheduler_state_dict` and the full
training `config`, to resume training; those saved by `train_image.py` carry `optimizer_state_dict`,
`config` and the `val_loss` they were selected with. The released ones don't.
"""

import inspect
import logging

import hydra
import torch

from models.base import InterpretableCLIP

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1


def checkpoint_dict(model: InterpretableCLIP, model_config: dict, backbone_config: dict, **extra) -> dict:
    """The checkpoint of `model`, built with the plain-dict kwargs `model_config` (`_target_` included).

    Every constructor default is written out, so the checkpoint doesn't depend on the defaults of the
    code that loads it; keys the constructor doesn't take are dropped.
    """
    params = inspect.signature(hydra.utils.get_class(model_config["_target_"]).__init__).parameters
    kwargs = {name: p.default for name, p in params.items() if p.default is not inspect.Parameter.empty}
    kwargs.update({k: v for k, v in model_config.items() if k in params})
    return {
        "format_version": FORMAT_VERSION,
        "model": {"_target_": model_config["_target_"], "features_size": model.features_size, **kwargs},
        "backbone": backbone_config,
        "text_concepts": list(model.text_concepts),
        "model_state_dict": model.state_dict(),
        **extra,
    }


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> InterpretableCLIP:
    """Load a GRACE checkpoint, in eval mode on `device`.

    The architecture comes from the checkpoint's own `model` kwargs, so the path alone is enough.
    """
    logger.info("Loading the checkpoint %s...", path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported checkpoint format {checkpoint.get('format_version')!r} (expected {FORMAT_VERSION})")

    state_dict = checkpoint["model_state_dict"]
    model = hydra.utils.instantiate(
        checkpoint["model"],
        text_concepts=checkpoint["text_concepts"],
        concepts_embds=state_dict["concepts.weight"],
    )
    model.load_state_dict(state_dict)
    return model.to(device).eval()

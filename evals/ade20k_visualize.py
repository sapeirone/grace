"""
Visualize where GRACE's top concepts are in ADE20k validation images (TIPSv2 backbone).

For each selected image, the top-k concepts of the image (CLS) embedding's reconstruction weights
are shown as heatmaps of their patch-level reconstruction weights, overlaid on the image (Figure 6
of the paper).
"""

import logging
import os

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import omegaconf  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from models.backbones.base import CLIP  # noqa: E402
from models.base import InterpretableCLIP  # noqa: E402
from models.checkpoint import load_checkpoint  # noqa: E402
from utils.entrypoint import log_banner  # noqa: E402

logger = logging.getLogger(__name__)

TITLE_FONTSIZE = 17
TEXT_COLOR = "black"


@torch.no_grad()
def top_concepts_figure(backbone: CLIP, model: InterpretableCLIP, dataset, indices: list[int], device: torch.device, top_k: int = 3):
    """One row per image: the image, then a heatmap of each of its top-k concepts.

    The top-k concepts come from the reconstruction weights of the (normalized) image embedding;
    each heatmap is that concept's reconstruction weight on every (unnormalized) patch embedding,
    min-max normalized per image, with transparency proportional to the activation.
    """
    cmap = plt.get_cmap("jet")
    fig, axes = plt.subplots(len(indices), top_k + 1, figsize=(3 * (top_k + 1), 3 * len(indices)), squeeze=False, facecolor="white")

    for row, idx in enumerate(indices):
        image, _, fn = dataset[idx]
        cls_embd, patches = backbone.encode_image(image[None].to(device), return_patches=True)
        _, weights = model.encode_image(F.normalize(cls_embd, dim=1), return_weights=True)
        _, patch_weights = model.encode_image(patches.reshape(-1, patches.shape[-1]), return_weights=True)
        patch_map = patch_weights.reshape(*patches.shape[1:3], -1)  # (grid, grid, num_concepts)

        top_scores, top_concepts = torch.topk(weights[0], top_k)

        # The image as read from disk (not the model input), resized to the input size.
        side = image.shape[-1]
        base_img = np.asarray(Image.open(dataset.root / "images" / dataset.split / fn).convert("RGB").resize((side, side))) / 255.0

        axes[row, 0].imshow(base_img, aspect="auto")
        axes[row, 0].axis("off")

        for col in range(top_k):
            concept_idx = top_concepts[col].item()
            heatmap = patch_map[:, :, concept_idx].float().cpu().numpy()
            # Per-image normalization, so weakly active areas stay transparent.
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap_rgba = cmap(heatmap_norm)
            heatmap_rgba[..., 3] = heatmap_norm  # alpha proportional to the activation

            ax = axes[row, col + 1]
            ax.imshow(base_img, aspect="auto")
            # Light dark veil, so the heatmap stands out on the image below.
            ax.imshow(np.zeros((*base_img.shape[:2], 4)) + [0, 0, 0, 0.35], aspect="auto")
            ax.imshow(heatmap_rgba, extent=(0, side, side, 0), interpolation="bilinear", aspect="auto")
            ax.set_title(f"{model.text_concepts[concept_idx]} ({top_scores[col].item():.3f})", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)
            ax.axis("off")

    plt.tight_layout()
    return fig


@hydra.main(config_path="../configs/experiments/", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_banner(logger)

    backbone = hydra.utils.instantiate(cfg.backbone).eval().to(device)
    model = load_checkpoint(cfg.resume_from, device)
    dataset = hydra.utils.instantiate(cfg.dataset, split="validation", transforms=backbone.preprocess)

    fig = top_concepts_figure(backbone, model, dataset, list(cfg.vis_indices), top_k=cfg.vis_top_k, device=device)

    output_path = cfg.vis_output or os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "top_concepts_val.jpg")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # The format follows the extension; JPEG keeps the (photo-like) figure ~10x smaller than PNG.
    pil_kwargs = {"quality": 90} if output_path.lower().endswith((".jpg", ".jpeg")) else None
    fig.savefig(output_path, facecolor="white", dpi=cfg.vis_dpi, bbox_inches="tight", pil_kwargs=pil_kwargs)
    logger.info("Saved the visualization to %s", output_path)


if __name__ == "__main__":
    main()

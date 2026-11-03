"""Image/caption preprocessing for the CLIP image experiments.

The image transform is the one the released image checkpoints were trained and evaluated with (a
bilinear resize), and what `CLIPBackbone.preprocess` returns; `clip.load`'s own (bicubic) transform
still runs, but the numbers no longer match the paper.
"""

import clip
import torch
import torchvision.transforms.v2 as v2

_CLIP_NORMALIZE = v2.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def get_clip_val_image_transform(input_size: int = 224) -> v2.Compose:
    """Get the (validation-time, deterministic) CLIP image preprocessing transform."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(input_size),
            v2.CenterCrop(input_size),
            v2.RGB(),
            v2.ToDtype(torch.float32, scale=True),
            _CLIP_NORMALIZE,
        ]
    )


def get_clip_val_caption_transform():
    """Get a transform that selects and tokenizes the first caption out of a list of captions
    (used for retrieval datasets, which return multiple captions per image)."""

    def transform(captions: list[str]) -> torch.Tensor:
        return clip.tokenize(captions[0], truncate=True)[0]

    return transform

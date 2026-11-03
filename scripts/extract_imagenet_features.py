"""Extract the CLIP ViT-B/32 features of ImageNet train used by `train_image.py`.

Reproduces the features `checkpoints/image/grace.pt` was trained on
(`<data root>/imagenet/train_features_clip_ViT-B-32.pt`): the
deterministic CLIP val transform (`data/transforms.py`), `clip.load("ViT-B/32")` on the GPU, i.e. in
fp16 as openai/CLIP loads it there, batches of 32 in `ImageFolder` order, and the raw (unnormalized)
image embeddings saved as float32 in a dict `{index in ImageFolder order: (512,) tensor}`.

Those features were extracted with torch 2.9.1 / torchvision 0.24.1, which this recipe reproduces bit
for bit. With the pinned torchvision 0.21 the resize of uint8 images differs slightly (cosine
similarity >= 0.998 to the original features).

Usage:
    python -m scripts.extract_imagenet_features
    python -m scripts.extract_imagenet_features --limit 2000 --output /tmp/check.pt   # a quick check
"""

import argparse
import os
from pathlib import Path

import clip
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from data.imagenet import ImageNetDataset
from data.transforms import get_clip_val_image_transform


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("GRACE_DATA", "grace_data")), help="data root (default: $GRACE_DATA, else grace_data/)")
    parser.add_argument("--split", default="train", help="ImageNet split directory under <root>/imagenet (default: train)")
    parser.add_argument("--output", type=Path, default=None, help="output file (default: <root>/imagenet/<split>_features_clip_ViT-B-32.pt)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size; 32 as for the released features (fp16 results depend slightly on it)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only extract the first N images (to check against the released features)")
    args = parser.parse_args()

    output = args.output or args.root / "imagenet" / f"{args.split}_features_clip_ViT-B-32.pt"
    dataset = ImageNetDataset(args.root / "imagenet", args.split, transform=get_clip_val_image_transform(224))
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model, _ = clip.load("ViT-B/32", device="cuda")
    model.eval()

    features = {}
    for images, _ in tqdm(loader, desc="Extracting ImageNet features"):
        embeddings = model.encode_image(images.to("cuda")).float().cpu()
        features.update(zip(range(len(features), len(features) + len(embeddings)), embeddings, strict=True))

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output)
    print(f"Saved {len(features)} features to {output}")


if __name__ == "__main__":
    main()

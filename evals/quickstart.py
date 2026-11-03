"""Decompose an image into GRACE concepts with the released CLIP ViT-B/32 checkpoint.

Usage: python -m evals.quickstart path/to/image.jpg [--checkpoint checkpoints/image/grace.pt] [--top-k 10]
"""

import argparse

import torch
from PIL import Image
from torch.nn import functional as F

from models.backbones.clip import CLIPBackbone
from models.checkpoint import load_checkpoint


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="path to the image to decompose")
    parser.add_argument("--checkpoint", default="checkpoints/image/grace.pt", help="GRACE checkpoint trained on CLIP ViT-B/32 features (default: %(default)s)")
    parser.add_argument("--top-k", type=int, default=10, help="number of concepts to print (default: %(default)s)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = CLIPBackbone().to(device).eval()
    grace = load_checkpoint(args.checkpoint, device)

    # CLIP embedding of the image
    image = backbone.preprocess(Image.open(args.image).convert("RGB"))
    features = backbone.encode_image(image[None].to(device))

    # Concept-based reconstruction of the embedding and its per-concept weights
    reconstructed, weights = grace.encode_image(features, return_weights=True)

    print(f"Cosine similarity between the CLIP embedding and its reconstruction: {F.cosine_similarity(reconstructed, features).item():.3f}")
    print(f"Top-{args.top_k} concepts:")
    top_weights, top_indices = weights[0].topk(args.top_k)
    for weight, index in zip(top_weights.tolist(), top_indices.tolist(), strict=True):
        print(f"  {grace.text_concepts[index]:30s} {weight:.3f}")


if __name__ == "__main__":
    main()

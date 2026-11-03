"""
ConViS-Bench, step 1: extract one feature file per video into `<out_video_dir>/<model_path>/<video_id>.pt`.

- `PE-Core-L14-336`: the raw Perception Encoder video embedding.
- `grace` / `splice`: the GRACE / SpLiCE reconstructed embedding.
- `grace_weights` / `splice_weights`: the GRACE / SpLiCE reconstruction weights.

Usage (from the repository root):
    python -m evals.convisbench.extract_features --model_path grace_weights --ckpt checkpoints/video/grace_full_vocab.pth
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evals.convisbench.video_dataset import VideoDataset
from models.backbones.pe import PEBackbone
from models.checkpoint import load_checkpoint
from models.splice import SpLiCE
from utils.concepts import load_concepts

# Default paths are relative to these files, so outputs stay under evals/convisbench/ whatever the CWD
CONVIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONVIS_DIR.parents[1]

MODELS = ["PE-Core-L14-336", "grace", "grace_weights", "splice", "splice_weights"]


def build_model(args, backbone: PEBackbone):
    """The concept-based model on top of `backbone` (None for the raw Perception Encoder)."""
    if args.model_path in ("grace", "grace_weights"):
        return load_checkpoint(args.ckpt, "cuda")
    if args.model_path in ("splice", "splice_weights"):
        text_concepts, concepts_embds = load_concepts(args.vocab, backbone)
        model = SpLiCE(
            features_size=backbone.features_size,
            text_concepts=text_concepts,
            concepts_embds=concepts_embds,
            l1_penalty=args.splice_l1_penalty,
            solver="admm",
        ).eval().cuda()
        if args.splice_centering:
            print("Using SpLiCE centering.")
            model.set_visual_mean(torch.load(REPO_ROOT / "data" / "means" / "egoclip_PE-Core-L14-336.pt", weights_only=True))
        return model
    return None


@torch.no_grad()
def main(args):
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    video_dataset = VideoDataset(args.video_dir, args.pairs_json, num_frames=args.num_frames)
    video_loader = DataLoader(video_dataset, batch_size=1, shuffle=False)
    output_dir = os.path.join(args.out_video_dir, args.model_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Extracting {args.model_path} features from {len(video_dataset)} videos to {output_dir}")

    backbone = PEBackbone(name="PE-Core-L14-336").eval().cuda()
    model = build_model(args, backbone)

    l0_norm = []
    for video_batch in tqdm(video_loader):
        video_id = video_batch["video_id"][0]
        video_frames = video_batch["video_frames"][0].permute(0, 3, 1, 2).unsqueeze(0)  # (B, T, C, H, W)
        video_features = backbone.encode_image(backbone.preprocess(video_frames).to("cuda"))

        if model is None:
            output = video_features.cpu()
        else:
            reconstructed, weights = model.encode_image(video_features, return_weights=True)
            output = weights if args.model_path.endswith("_weights") else reconstructed
            if args.model_path.startswith("splice"):
                l0_norm.append(torch.count_nonzero(weights, dim=1).float().item())
        torch.save(output, os.path.join(output_dir, video_id + ".pt"))

    if l0_norm:
        print("Average L0 norm of the SpLiCE weights:", np.mean(l0_norm))
    print(f"Visual features of {len(video_dataset)} videos saved to {output_dir}")


if __name__ == "__main__":
    # ConViS-Bench in the data root (scripts/download_data.py)
    CONVISBENCH_DATA = os.path.join(os.environ.get("GRACE_DATA", REPO_ROOT / "grace_data"), "convisbench")
    parser = argparse.ArgumentParser(description="Extract the visual features of the ConViS-Bench videos.")
    parser.add_argument("--model_path", type=str, required=True, choices=MODELS, help="Model (one row of Table 1).")
    parser.add_argument("--ckpt", type=str, default="", help="GRACE checkpoint (grace / grace_weights).")
    parser.add_argument(
        "--vocab",
        type=str,
        default=str(REPO_ROOT / "data" / "concepts_vocs" / "video" / "egoclip_mined_top25_verbs_nouns.txt"),
        help="SpLiCE concept vocabulary.",
    )
    # `type=bool`: any non-empty value (even "False") turns it on
    parser.add_argument("--splice-centering", type=bool, default=True, help="Center SpLiCE on the EgoClip visual mean.")
    parser.add_argument("--splice-l1-penalty", type=float, default=0.15, help="SpLiCE's L1 penalty (λ).")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames sampled uniformly from each video.")
    parser.add_argument("--video_dir", type=str, default=CONVISBENCH_DATA, help="Directory of the ConViS-Bench videos.")
    parser.add_argument("--pairs_json", type=str, default=os.path.join(CONVISBENCH_DATA, "ConVIS.jsonl"), help="JSONL of the video pairs.")
    parser.add_argument("--out_video_dir", type=str, default=str(CONVIS_DIR / "visual_features"), help="Where the features go.")
    main(parser.parse_args())

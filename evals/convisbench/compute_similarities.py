"""
ConViS-Bench, step 2: the similarity of every video pair, from the features of step 1, saved to
`<similarity_dir>/<model_path>.json`.

- `PE-Core-L14-336` / `grace` / `splice` (CLIPScore): cosine similarity between one video's embedding
  and the Perception Encoder text embedding of the other video's caption, averaged over both
  directions. The captions are mPLUG-Owl3's (`captions/mPLUG-Owl3-7B-240728/`).
- `grace_weights`: negative Jensen–Shannon distance between the two videos' GRACE reconstruction weights.
- `splice_weights`: negative mean absolute difference between the two videos' SpLiCE weights.

Usage (from the repository root):
    python -m evals.convisbench.compute_similarities --model_path grace_weights
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

from models.backbones.pe import PEBackbone

# Default paths are relative to these files, so outputs stay under evals/convisbench/ whatever the CWD
CONVIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONVIS_DIR.parents[1]

CLIPSCORE_MODELS = ["PE-Core-L14-336", "grace", "splice"]
MODELS = CLIPSCORE_MODELS + ["grace_weights", "splice_weights"]


def video_id(path: str) -> str:
    return path.split("/")[-1].split(".")[0]


def jsd_similarity(weights1: torch.Tensor, weights2: torch.Tensor) -> float:
    mixture = 0.5 * (weights1 + weights2) + 1e-8
    jsd = 0.5 * (
        F.kl_div((weights1 + 1e-8).log(), mixture, reduction="batchmean")
        + F.kl_div((weights2 + 1e-8).log(), mixture, reduction="batchmean")
    )
    return -jsd.sqrt().item()


def l1_similarity(weights1: torch.Tensor, weights2: torch.Tensor) -> float:
    return -torch.abs(weights1.squeeze() - weights2.squeeze()).mean().item()


@torch.no_grad()
def main(args):
    with open(args.pairs_jsonl, "r") as f:
        pairs = [json.loads(line) for line in f]
    load = lambda path: torch.load(os.path.join(args.feature_dir, args.model_path, f"{video_id(path)}.pt"), weights_only=True)

    if args.model_path in CLIPSCORE_MODELS:
        text_encoder = PEBackbone(name="PE-Core-L14-336").eval().cuda()
        print(f"Loading the captions from {args.captions_dir}")

        def caption_features(path: str) -> torch.Tensor:
            with open(os.path.join(args.captions_dir, f"{video_id(path)}.txt"), "r") as f:
                return text_encoder.encode_text(f.read().strip())

    similarities = []
    for pair in tqdm(pairs):
        features1, features2 = load(pair["video1"]), load(pair["video2"])
        if args.model_path == "grace_weights":
            similarity = jsd_similarity(features1, features2)
        elif args.model_path == "splice_weights":
            similarity = l1_similarity(features1, features2)
        else:
            features1, features2 = features1.to("cuda"), features2.to("cuda")
            video2text = F.cosine_similarity(features1, caption_features(pair["video2"]), dim=1, eps=1e-6).item()
            text2video = F.cosine_similarity(features2, caption_features(pair["video1"]), dim=1, eps=1e-6).item()
            similarity = ((video2text + text2video) / 2 + 1) / 2
        # 7 significant digits: fewer would create ties between pairs and change the rank correlations
        # (6 digits already changes one Kendall τ of the Perception Encoder row)
        similarities.append({"video1": pair["video1"], "video2": pair["video2"], "similarity": float(f"{similarity:.7g}")})

    os.makedirs(args.similarity_dir, exist_ok=True)
    output_file = os.path.join(args.similarity_dir, f"{args.model_path}.json")
    with open(output_file, "w") as f:
        json.dump(similarities, f, indent=4)
    print(f"Similarities saved to {output_file}")


if __name__ == "__main__":
    # ConViS-Bench in the data root (scripts/download_data.py)
    CONVISBENCH_DATA = os.path.join(os.environ.get("GRACE_DATA", REPO_ROOT / "grace_data"), "convisbench")
    parser = argparse.ArgumentParser(description="Compute the similarity of every ConViS-Bench video pair.")
    parser.add_argument("--model_path", type=str, required=True, choices=MODELS, help="Model (one row of Table 1).")
    parser.add_argument("--feature_dir", type=str, default=str(CONVIS_DIR / "visual_features"), help="Features of step 1.")
    parser.add_argument("--captions_dir", type=str, default=str(CONVIS_DIR / "captions" / "mPLUG-Owl3-7B-240728"), help="Video captions (CLIPScore).")
    parser.add_argument("--pairs_jsonl", type=str, default=os.path.join(CONVISBENCH_DATA, "ConVIS.jsonl"), help="JSONL of the video pairs.")
    parser.add_argument("--similarity_dir", type=str, default=str(CONVIS_DIR / "computed_similarities"), help="Where the similarities go.")
    main(parser.parse_args())

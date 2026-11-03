"""Build the decomposition examples shown on the project page (`docs/assets/examples/`).

Run from the repo root with the repo's environment (checkpoints and `$GRACE_DATA` as in CLAUDE.md):

    python docs/build_examples.py                    # the hand-picked examples (the *_IDS lists below)
    python docs/build_examples.py --random --seed 0  # uniform random samples, not filtered

Images: COCO val2017, `checkpoints/image/grace.pt` (CLIP ViT-B/32).
Egocentric videos: EgoMCQ answer clips (precomputed PE features), `checkpoints/video/grace.pth`.
ConVisBench videos: the GRACE weights extracted by `evals.convisbench.extract_features --model_path
grace_weights` (`grace_full_vocab.pth`, as in Table 1 of the paper).
The video tabs also get short, muted MP4s (ffmpeg, libx264): the whole ConVisBench video, sped up to at most
`MAX_VIDEO_SECONDS`, and each Ego4D clip within `EGO_CONTEXT_SECONDS` of context.
Visual grounding: ADE20k validation images, `checkpoints/image/grace_tipsv2_l14_laion.pt` (TIPSv2-L14),
heatmaps as in `evals/ade20k_visualize.py` (Figure 6 of the paper).

Pass group names to rebuild only those (the others are kept from the current `examples.js`):

    python docs/build_examples.py grounding
"""

import argparse
import json
import logging
import os
import os.path as osp
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = os.getcwd()  # run from the repo root
sys.path.insert(0, REPO)
logging.disable(logging.CRITICAL)

from data.egoclip import load_frames  # noqa: E402
from data.egomcq import EgoMCQDataset  # noqa: E402
from models.backbones.clip import CLIPBackbone  # noqa: E402
from models.checkpoint import load_checkpoint  # noqa: E402

DATA = os.environ.get("GRACE_DATA", osp.join(REPO, "grace_data"))
CKPTS = osp.join(REPO, "checkpoints")
CONVIS = osp.join(REPO, "evals", "convisbench")
OUT = osp.join(osp.dirname(osp.abspath(__file__)), "assets", "examples")
TOP_K = 5
NUM_EXAMPLES = 6  # per group, with --random
MAX_VIDEO_SECONDS = 12  # longer ConVisBench videos are sped up to fit
EGO_CONTEXT_SECONDS = 6  # Ego4D clips last about a second: show them within this much context

# Hand-picked examples (the default selection)
COCO_IDS = [94852, 235057, 473121, 1584, 251140, 573094]
EGOMCQ_IDS = [14529, 34167, 37192, 17071, 38232, 31234]  # indices into EgoMCQDataset.samples
CONVIS_IDS = ["1692_001", "1754_005", "5854_000", "121_003", "4762_011", "2722_000"]
ADE20K_IDS = [228, 585, 1699]  # validation indices in sorted filename order, 228 and 585 are in Figure 6
GROUNDING_TOP_K = 3
# Concept shown first on the page (index into the top-k), where an object concept localizes better than the scene-level top one.
# Random samples always start from the top concept.
GROUNDING_DEFAULT = {585: 1, 1699: 1}


def top_concepts(weights: torch.Tensor, concepts: list[str]) -> list[list]:
    values, indices = weights.topk(TOP_K)
    return [[concepts[i], round(v, 4)] for v, i in zip(values.tolist(), indices.tolist(), strict=True)]


def save_thumb(image: Image.Image, path: str, max_side: int = 520) -> str:
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    image.save(path, quality=82)
    return osp.relpath(path, osp.dirname(osp.dirname(OUT)))  # relative to docs/


def video_frames(path: str, indices: list[int]) -> list[Image.Image]:
    import decord

    reader = decord.VideoReader(path)
    indices = [min(max(i, 0), len(reader) - 1) for i in indices]
    frames = reader.get_batch(indices)
    frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()  # the decord bridge may be torch
    return [Image.fromarray(frame) for frame in frames]


def encode_video(src: str, name: str, start: float, duration: float, speed: float = 1.0, fps: float | None = None) -> dict:
    """Encode `[start, start + duration)` of `src` as a small, muted H.264 MP4 (`speed`x faster), plus a poster frame."""
    path, poster = osp.join(OUT, f"{name}.mp4"), osp.join(OUT, f"{name}_poster.jpg")
    filters = f"setpts=PTS/{speed},scale=480:-2" + (f",fps={fps}" if fps else "")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", src, "-an", "-vf", filters,
                    "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-frames:v", "1", "-q:v", "4", poster], check=True)
    website = osp.dirname(osp.dirname(OUT))
    return {"src": osp.relpath(path, website), "poster": osp.relpath(poster, website), "speed": round(speed, 1)}


def pick(curated: list, population: list, seed: int | None) -> list:
    """The curated ids, or (with a seed) `NUM_EXAMPLES` uniform random ones from `population`."""
    return curated if seed is None else random.Random(seed).sample(population, NUM_EXAMPLES)


@torch.no_grad()
def images(seed: int | None) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = CLIPBackbone().to(device).eval()
    grace = load_checkpoint(osp.join(CKPTS, "image", "grace.pt"), device).eval()
    captions = {}
    for ann in json.load(open(osp.join(DATA, "coco", "annotations", "captions_val2017.json")))["annotations"]:
        captions.setdefault(ann["image_id"], ann["caption"].strip())

    examples = []
    for image_id in pick(COCO_IDS, sorted(captions), seed):
        image = Image.open(osp.join(DATA, "coco", "val2017", f"{image_id:012d}.jpg")).convert("RGB")
        features = backbone.encode_image(backbone.preprocess(image)[None].to(device))
        _, weights = grace.encode_image(features, return_weights=True)
        examples.append({
            "id": f"coco-{image_id}",
            "source": f"COCO val2017 #{image_id}",
            "caption": captions[image_id],
            "media": [save_thumb(image, osp.join(OUT, f"coco_{image_id}.jpg"))],
            "concepts": top_concepts(weights[0].cpu(), grace.text_concepts),
        })
    return examples


@torch.no_grad()
def egocentric(seed: int | None) -> list[dict]:
    dataset = EgoMCQDataset(
        osp.join(DATA, "ego4d", "annotations"), osp.join(DATA, "ego4d", "features", "EgoMCQ_PE-Core-L14-336"), osp.join(DATA, "ego4d_videos"), False, None
    )
    grace = load_checkpoint(osp.join(CKPTS, "video", "grace.pth"), "cpu").eval()
    # A few questions have no precomputed features: random samples skip them
    candidates = EGOMCQ_IDS if seed is None else random.Random(seed).sample(range(len(dataset)), len(dataset))

    examples = []
    for idx in candidates:
        sample = dataset.samples[idx]
        video_features, _, text, metadata = dataset[idx]
        if metadata["answer"] < 0:
            continue
        _, weights = grace.encode_image(video_features[metadata["answer"]][None], return_weights=True)
        video_uid, start, end = sample.choices[metadata["answer"]]
        # `start` / `end` are 30 fps frame indices, `load_frames` maps them onto (possibly subsampled) videos.
        # The clips last about a second, so a single middle frame shows them best.
        video_path = osp.join(DATA, "ego4d_videos", f"{video_uid}.mp4")
        frame = load_frames(video_path, start, end, num_frames=3)[1]
        # The clip (in seconds) within a window of context, which the page plays with the clip marked on a timeline
        import decord

        reader = decord.VideoReader(video_path)
        clip_start, clip_end = start / 30, end / 30
        window_start = max(0.0, min((clip_start + clip_end - EGO_CONTEXT_SECONDS) / 2, len(reader) / reader.get_avg_fps() - EGO_CONTEXT_SECONDS))
        video = encode_video(video_path, f"ego_{idx}", window_start, EGO_CONTEXT_SECONDS)
        video["clip"] = [round(clip_start - window_start, 2), round(clip_end - window_start, 2)]
        narration = text.replace("#c c ", "C ").replace("#o ", "").strip().rstrip(".")
        examples.append({
            "id": f"egomcq-{idx}",
            "source": f"Ego4D · EgoMCQ question {sample.ann_id} · {(end - start) / 30:.1f} s clip",
            "caption": narration[0].upper() + narration[1:] + ".",
            "media": [save_thumb(Image.fromarray(frame.permute(1, 2, 0).numpy()), osp.join(OUT, f"ego_{idx}.jpg"))],
            "video": video,
            "concepts": top_concepts(weights[0], grace.text_concepts),
        })
        if len(examples) == NUM_EXAMPLES:
            break
    return examples


def short_caption(caption: str) -> str:
    """The first sentence of an mPLUG-Owl3 caption, without its "The video shows ..." boilerplate (the page shows one line)."""
    first = caption.split(". ")[0].rstrip(".")
    for prefix in ("The video shows ", "The video features ", "This video shows ", "In the video, ", "In this video, "):
        if first.startswith(prefix):
            first = first.removeprefix(prefix)
            break
    first = first.replace(" is seen ", " ").replace(" are seen ", " ")
    return first[0].upper() + first[1:] + "."


def convisbench(seed: int | None) -> list[dict]:
    import decord

    concepts = torch.load(osp.join(CKPTS, "video", "grace_full_vocab.pth"), weights_only=True, map_location="cpu")["text_concepts"]
    videos = sorted(f.removesuffix(".pt") for f in os.listdir(osp.join(CONVIS, "visual_features", "grace_weights")))
    examples = []
    for video_id in pick(CONVIS_IDS, videos, seed):
        weights = torch.load(osp.join(CONVIS, "visual_features", "grace_weights", f"{video_id}.pt"))[0]
        caption = open(osp.join(CONVIS, "captions", "mPLUG-Owl3-7B-240728", f"{video_id}.txt")).read().strip()
        path = osp.join(DATA, "convisbench", "videos", f"{video_id}.mp4")
        reader = decord.VideoReader(path)
        length, duration = len(reader), len(reader) / reader.get_avg_fps()
        frames = video_frames(path, [round((length - 1) * t) for t in (0.1, 0.37, 0.63, 0.9)])
        video = encode_video(path, f"convis_{video_id}", 0.0, duration, speed=max(1.0, duration / MAX_VIDEO_SECONDS), fps=24)
        examples.append({
            "id": f"convis-{video_id}",
            "source": f"ConVisBench · {video_id}",
            "caption": short_caption(caption),
            "media": [save_thumb(frame, osp.join(OUT, f"convis_{video_id}_{j}.jpg"), 360) for j, frame in enumerate(frames)],
            "video": video,
            "concepts": top_concepts(weights, concepts),
        })
    return examples


def heatmap_overlay(image: np.ndarray, heatmap: np.ndarray) -> Image.Image:
    """`evals/ade20k_visualize.py`'s overlay: the heatmap, min-max normalized per image, in the jet colormap
    with alpha proportional to the activation, over the image under a light dark veil."""
    from matplotlib import colormaps

    side = image.shape[0]
    norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    norm = np.clip(np.asarray(Image.fromarray(norm.astype(np.float32), mode="F").resize((side, side), Image.BILINEAR)), 0, 1)
    alpha = norm[..., None]
    overlay = image * 0.65 * (1 - alpha) + colormaps["jet"](norm)[..., :3] * alpha
    return Image.fromarray((overlay * 255).round().astype(np.uint8))


@torch.no_grad()
def grounding(seed: int | None) -> list[dict]:
    from data.ade20k import ADE20KDataset
    from models.backbones.tipsv2 import TIPSv2Backbone

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = TIPSv2Backbone().eval().to(device)
    grace = load_checkpoint(osp.join(CKPTS, "image", "grace_tipsv2_l14_laion.pt"), device).eval()
    dataset = ADE20KDataset(osp.join(DATA, "ade20k", "ADEChallengeData2016"), "validation", transforms=backbone.preprocess)

    examples = []
    for idx in pick(ADE20K_IDS, list(range(len(dataset))), seed):
        image, _, fn = dataset[idx]
        # Top concepts from the (normalized) image embedding, heatmaps from the (unnormalized) patch embeddings
        cls_embd, patches = backbone.encode_image(image[None].to(device), return_patches=True)
        _, weights = grace.encode_image(F.normalize(cls_embd, dim=1), return_weights=True)
        _, patch_weights = grace.encode_image(patches.reshape(-1, patches.shape[-1]), return_weights=True)
        patch_map = patch_weights.reshape(*patches.shape[1:3], -1).float().cpu().numpy()  # (grid, grid, num_concepts)
        top_weights, top_indices = weights[0].topk(GROUNDING_TOP_K)

        side = image.shape[-1]
        base = np.asarray(Image.open(dataset.root / "images" / dataset.split / fn).convert("RGB").resize((side, side))) / 255.0
        examples.append({
            "id": f"ade20k-{idx}",
            "source": f"ADE20k validation · {fn.removesuffix('.jpg')}",
            "media": [save_thumb(Image.fromarray((base * 255).round().astype(np.uint8)), osp.join(OUT, f"ade_{idx}.jpg"), side)],
            "overlays": [save_thumb(heatmap_overlay(base, patch_map[:, :, c]), osp.join(OUT, f"ade_{idx}_{k}.jpg"), side) for k, c in enumerate(top_indices.tolist())],
            "concepts": [[grace.text_concepts[c], round(w, 4)] for w, c in zip(top_weights.tolist(), top_indices.tolist(), strict=True)],
            "default": GROUNDING_DEFAULT.get(idx, 0) if seed is None else 0,
        })
    return examples


GROUPS = {"images": images, "egocentric": egocentric, "convisbench": convisbench, "grounding": grounding}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="*", help=f"groups to rebuild, among {', '.join(GROUPS)} (default: all)")
    parser.add_argument("--random", action="store_true", help="uniform random samples instead of the hand-picked ones")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random samples (default: %(default)s)")
    args = parser.parse_args()
    if unknown := set(args.groups) - set(GROUPS):
        parser.error(f"unknown groups: {', '.join(sorted(unknown))}")
    seed = args.seed if args.random else None

    os.makedirs(OUT, exist_ok=True)
    path = osp.join(osp.dirname(OUT), "examples.js")
    data = {}
    if args.groups and osp.exists(path):
        data = json.loads(open(path).read().split("=", 1)[1].strip().removesuffix(";"))
    meta = data.pop("meta", {})
    for group in args.groups or GROUPS:
        data[group] = GROUPS[group](seed)
        meta[group] = {"selection": "curated"} if seed is None else {"selection": "random", "seed": seed}
    data["meta"] = meta
    with open(path, "w") as f:
        f.write("// Generated by docs/build_examples.py, do not edit by hand.\n")
        f.write(f"window.GRACE_EXAMPLES = {json.dumps(data, indent=1, ensure_ascii=False)};\n")
    for group in GROUPS:
        print(group, meta.get(group))
        for example in data.get(group, []):
            print(f"  {example['source']:40s} {example.get('caption', '')[:60]:60s} {', '.join(c for c, _ in example['concepts'])}")


if __name__ == "__main__":
    main()

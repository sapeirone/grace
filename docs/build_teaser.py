"""Build the looped teaser video of the project page (`docs/assets/img/teaser_<set>.mp4`).

Three EgoMCQ clips, one after the other: the video on the left, GRACE's top concepts (`checkpoints/video/grace.pth`)
growing in as bars on the right, with a crossfade between clips. Run from the repo root (as `build_examples.py`):

    python docs/build_teaser.py                # the crafts set (the page's teaser): ironing, cutting cloth, splitting coconuts
    python docs/build_teaser.py --set cooking  # cracking eggs, mixing flour, stoking the fire under a stove
"""

import argparse
import logging
import os
import os.path as osp
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = os.getcwd()  # run from the repo root
sys.path.insert(0, REPO)
logging.disable(logging.CRITICAL)

from data.egoclip import load_frames  # noqa: E402
from data.egomcq import EgoMCQDataset  # noqa: E402
from models.checkpoint import load_checkpoint  # noqa: E402

DATA = os.environ.get("GRACE_DATA", osp.join(REPO, "grace_data"))
OUT = osp.join(osp.dirname(osp.abspath(__file__)), "assets", "img")

# Indices into EgoMCQDataset.samples
SETS = {
    "cooking": [21982, 9775, 20900],  # cracking eggs, mixing flour, stoking the fire under a stove
    "crafts": [36238, 27522, 14529],  # ironing a shirt, cutting cloth, splitting coconuts
}
TOP_K = 4
FPS = 25
SECONDS = 5.0  # per clip, including the crossfade into the next one
FADE = 0.5
CONTEXT = 5.0  # seconds of video around each (~1 s) narrated clip

S = 2.0  # render scale: 2560x720 output
W, H = int(1280 * S), int(360 * S)
BG = (247, 250, 248)
INK, MUTED, GREEN, TRACK = (21, 35, 29), (88, 107, 98), (47, 138, 95), (226, 234, 229)
FONTS = "/usr/share/fonts/truetype"


def s(x: float) -> int:
    return round(x * S)


# A 4:3 video panel on the left, the text on the right
VIDEO_BOX = (s(30), s(30), s(30) + s(400), s(30) + s(300))
TEXT_X, TEXT_W = s(474), s(766)
LINE, HEAD, ROW = s(32), s(34), s(44)


def font(name: str, size: float) -> ImageFont.FreeTypeFont:
    path = f"{FONTS}/dejavu/{name}" if name.startswith("DejaVu") else f"{FONTS}/lato/{name}"
    return ImageFont.truetype(path, round(size * S))


F_NARR, F_HEAD = font("Lato-MediumItalic.ttf", 25), font("Lato-Bold.ttf", 17)
F_CONCEPT, F_WEIGHT = font("Lato-Semibold.ttf", 21), font("DejaVuSansMono.ttf", 16)


def ease_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def rounded(image: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *image.size), radius, fill=255)
    out = Image.new("RGB", image.size, BG)
    out.paste(image, (0, 0), mask)
    return out


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=fnt) <= width or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    return lines + [line]


@torch.no_grad()
def load_examples(ids: list[int]) -> list[dict]:
    dataset = EgoMCQDataset(
        osp.join(DATA, "ego4d", "annotations"), osp.join(DATA, "ego4d", "features", "EgoMCQ_PE-Core-L14-336"), osp.join(DATA, "ego4d_videos"), False, None
    )
    grace = load_checkpoint(osp.join(REPO, "checkpoints", "video", "grace.pth"), "cpu").eval()
    size = (VIDEO_BOX[2] - VIDEO_BOX[0], VIDEO_BOX[3] - VIDEO_BOX[1])
    examples = []
    for idx in ids:
        sample = dataset.samples[idx]
        video_features, _, text, metadata = dataset[idx]
        _, weights = grace.encode_image(video_features[metadata["answer"]][None], return_weights=True)
        values, indices = weights[0].topk(TOP_K)
        video_uid, start, end = sample.choices[metadata["answer"]]
        # CONTEXT seconds around the clip (30 fps indices), one frame per 0.2 s (the local copies are 5 fps)
        middle, half = (start + end) // 2, round(CONTEXT * 30 / 2)
        frames = load_frames(osp.join(DATA, "ego4d_videos", f"{video_uid}.mp4"), max(0, middle - half), middle + half, num_frames=round(CONTEXT * 5))
        narration = " ".join(text.replace("#c c ", "C ").replace("#o ", "").split()).rstrip(".")
        examples.append({
            "narration": "“" + narration[0].upper() + narration[1:] + ".”",
            "frames": [rounded(Image.fromarray(f.permute(1, 2, 0).numpy()).convert("RGB").resize(size, Image.LANCZOS), s(16)) for f in frames],
            "concepts": [(grace.text_concepts[i], v) for v, i in zip(values.tolist(), indices.tolist(), strict=True)],
        })
        print(idx, examples[-1]["narration"], [c for c, _ in examples[-1]["concepts"]])
    return examples


def render(examples: list[dict], k: int, t: float) -> Image.Image:
    """Frame of example `k` at time `t` within its segment."""
    ex = examples[k]
    image = Image.new("RGB", (W, H), BG)
    frame = ex["frames"][min(int(t / CONTEXT * len(ex["frames"])), len(ex["frames"]) - 1)]
    image.paste(frame, VIDEO_BOX[:2])

    draw = ImageDraw.Draw(image)
    x = TEXT_X
    # Same slots for every example (as many narration lines as the longest one), centred on the video
    lines = max(len(wrap(draw, e["narration"], F_NARR, TEXT_W)) for e in examples)
    block = lines * LINE + s(12) + HEAD + len(ex["concepts"]) * ROW + s(24)
    top = VIDEO_BOX[1] + (VIDEO_BOX[3] - VIDEO_BOX[1] - block) // 2
    for n, line in enumerate(wrap(draw, ex["narration"], F_NARR, TEXT_W)):
        draw.text((x, top + n * LINE), line, font=F_NARR, fill=INK)
    y = top + lines * LINE + s(12)
    draw.text((x, y), "GRACE concept decomposition", font=F_HEAD, fill=MUTED)
    y += HEAD

    max_weight = ex["concepts"][0][1]
    bar_x0, bar_x1 = x + s(160), x + TEXT_W - s(84)
    for i, (concept, weight) in enumerate(ex["concepts"]):
        grow = ease_out((t - 0.35 - 0.09 * i) / 0.9)
        cy = y + s(18)
        draw.text((x, cy), concept, font=F_CONCEPT, fill=INK, anchor="lm")
        draw.rounded_rectangle((bar_x0, cy - s(7), bar_x1, cy + s(7)), s(7), fill=TRACK)
        if grow > 0:
            fill_x = bar_x0 + max(s(14), (bar_x1 - bar_x0) * weight / max_weight * grow)
            draw.rounded_rectangle((bar_x0, cy - s(7), fill_x, cy + s(7)), s(7), fill=GREEN)
        if grow > 0.6:
            draw.text((x + TEXT_W, cy), f"{weight:.3f}", font=F_WEIGHT, fill=MUTED, anchor="rm")
        y += ROW

    # progress: one dot per example
    for j in range(len(examples)):
        cx, cy = x + s(7) + j * s(24), y + s(12)
        draw.ellipse((cx - s(5), cy - s(5), cx + s(5), cy + s(5)), fill=GREEN if j == k else TRACK)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=list(SETS), default="crafts", help="example set (default: %(default)s)")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    examples = load_examples(SETS[args.set])
    frames_per_example = round(SECONDS * FPS)
    path = osp.join(OUT, f"teaser_{args.set}.mp4")
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
        stdin=subprocess.PIPE,
    )
    for k in range(len(examples)):
        for n in range(frames_per_example):
            t = n / FPS
            image = render(examples, k, t)
            if t > SECONDS - FADE:  # into the next example's first frame (the last one into the first: seamless loop)
                alpha = (t - (SECONDS - FADE)) / FADE
                following = render(examples, (k + 1) % len(examples), 0.0)
                video = Image.blend(image, following, alpha).crop(VIDEO_BOX)
                blank = Image.new("RGB", (W, H), BG)
                image = Image.blend(image, blank, alpha * 2) if alpha < 0.5 else Image.blend(blank, following, alpha * 2 - 1)
                image.paste(video, VIDEO_BOX[:2])
            if k == 0 and n == 0:
                image.save(osp.join(OUT, f"teaser_{args.set}_poster.jpg"), quality=88)
            ffmpeg.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

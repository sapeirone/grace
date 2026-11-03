"""Extract the Perception Encoder (PE-Core-L14-336) features of EgoClip / EgoMCQ from the Ego4D videos.

Produces the `features.pt` read by `data/egoclip.py` / `data/egomcq.py`, with the recipe of the released
features: 16 frames evenly sampled from each clip, `PEBackbone.preprocess`, video and narration
embeddings in fp32.

- EgoClip: `features[video_uid][narration_source] = {"video_features": (N, 1024), "text_features":
  (N, 1024)}` in float16, row `narration_ind` for each narration of `egoclip.csv` (N = the largest
  index + 1; rows without a narration are zero).
- EgoMCQ: `features[question_id] = (video_features (5, 1024), text_features (1024,))` in float32, one
  row per candidate clip.

Videos missing from `videos_root` are skipped (Ego4D v2 dropped a few), as are the EgoMCQ questions
that use them. The features of each video are saved to `<output_dir>/shards/<video_uid>.pt` as soon
as it's done, so an interrupted run resumes where it stopped, and `num_slices` / `slice` split the
videos across parallel jobs. Once every shard is there, the run that finishes last merges them into
`<output_dir>/features.pt` (EgoClip's takes ~16 GB of RAM); point the dataset's `features_root` to
`<output_dir>`, or extract to the default location
(`$GRACE_DATA/ego4d/features/{EgoClip,EgoMCQ}_PE-Core-L14-336`).

Usage:
    python -m scripts.extract_ego4d_features dataset=egomcq dataset.videos_root=/path/to/videos \\
        output_dir=$GRACE_DATA/ego4d/features/EgoMCQ_PE-Core-L14-336
    python -m scripts.extract_ego4d_features dataset=egoclip dataset.videos_root=/path/to/videos \\
        output_dir=$GRACE_DATA/ego4d/features/EgoClip_PE-Core-L14-336 [num_slices=8 slice=0]
"""

import logging
import os
import os.path as osp
from collections import defaultdict

import einops
import hydra
import omegaconf
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from data.egoclip import EgoClipDataset
from data.egomcq import EgoMCQDataset
from models.backbones.pe import PEBackbone
from utils.entrypoint import log_banner

logger = logging.getLogger(__name__)


def sample_videos(dataset: EgoClipDataset | EgoMCQDataset, idx: int) -> tuple[str, set[str]]:
    """The shard (video) a sample belongs to, and the videos it reads."""
    sample = dataset.samples[idx]
    if isinstance(dataset, EgoMCQDataset):
        return sample.query_video_uid, {video_uid for video_uid, *_ in sample.choices}
    return sample.video_uid, {sample.video_uid}


def build_shard(dataset: EgoClipDataset | EgoMCQDataset, indices: list[int], video_features: torch.Tensor, text_features: torch.Tensor) -> dict:
    """The features of one video's samples, in the layout of `features.pt`."""
    if isinstance(dataset, EgoMCQDataset):
        return {dataset.samples[i].ann_id: (v, t) for i, v, t in zip(indices, video_features, text_features, strict=True)}

    shard = {}
    for source in sorted({dataset.samples[i].narration_source for i in indices}):
        rows = [(dataset.samples[i].narration_ind, v, t) for i, v, t in zip(indices, video_features, text_features, strict=True) if dataset.samples[i].narration_source == source]
        num_rows = max(ind for ind, *_ in rows) + 1
        shard[source] = {"video_features": torch.zeros(num_rows, video_features.shape[-1], dtype=torch.float16), "text_features": torch.zeros(num_rows, text_features.shape[-1], dtype=torch.float16)}
        for ind, v, t in rows:
            shard[source]["video_features"][ind], shard[source]["text_features"][ind] = v, t
    return shard


def encode(model: PEBackbone, frames: torch.Tensor, texts: list[str], is_mcq: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Video and text embeddings of a batch (EgoMCQ: frames are `(B, 5, T, C, H, W)`, one row per candidate clip)."""
    if is_mcq:
        video_features = einops.rearrange(model.encode_image(einops.rearrange(frames, "B N T C H W -> (B N) T C H W")), "(B N) F -> B N F", B=frames.shape[0])
    else:
        video_features = model.encode_image(frames)
    return video_features, model.encode_text(texts)


@hydra.main(config_path="../configs/experiments/features_extraction", config_name="pe", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:
    """Extract the backbone features of one slice of the dataset's videos and merge them once all slices are done.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra config
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_banner(logger)

    model = hydra.utils.instantiate(cfg.model).to(device).eval()
    dataset = hydra.utils.instantiate(cfg.dataset, ret_videos=True, transforms=model.preprocess)
    is_mcq = isinstance(dataset, EgoMCQDataset)

    # Group the samples by video (EgoMCQ: by the query's video), skipping the ones that read missing videos
    available = {f.removesuffix(".mp4") for f in os.listdir(dataset.videos_root) if f.endswith(".mp4")}
    groups, num_skipped = defaultdict(list), 0
    for idx in range(len(dataset)):
        shard_uid, videos = sample_videos(dataset, idx)
        if videos <= available:
            groups[shard_uid].append(idx)
        else:
            num_skipped += 1
    shard_uids = sorted(groups)
    logger.info("%d samples from %d videos (%d samples skipped: missing videos)", sum(map(len, groups.values())), len(shard_uids), num_skipped)

    shards_dir = osp.join(cfg.output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    # This job's slice of the videos, without those already done
    if cfg.num_slices > 0:
        shard_uids_slice = shard_uids[len(shard_uids) * cfg.slice // cfg.num_slices : len(shard_uids) * (cfg.slice + 1) // cfg.num_slices]
    else:
        shard_uids_slice = shard_uids
    todo = [uid for uid in shard_uids_slice if not osp.exists(osp.join(shards_dir, f"{uid}.pt"))]
    logger.info("Extracting %d videos (%d already done)...", len(todo), len(shard_uids_slice) - len(todo))

    indices = [idx for uid in todo for idx in groups[uid]]
    clips_per_sample = 5 if is_mcq else 1
    loader = DataLoader(Subset(dataset, indices), batch_size=max(1, cfg.batch_size // clips_per_sample), shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # The loader follows `indices`, i.e. one video after the other: save each video's shard once all its samples are in
    pending_video, pending_text = [], []
    remaining = iter(todo)
    current_uid = next(remaining, None)
    for frames, texts, _ in tqdm(loader, desc="Extracting features"):
        with torch.inference_mode():
            video_features, text_features = encode(model, frames.to(device), list(texts), is_mcq)
        pending_video += list(video_features.float().cpu())
        pending_text += list(text_features.float().cpu())

        while current_uid is not None and len(pending_video) >= len(groups[current_uid]):
            n = len(groups[current_uid])
            shard = build_shard(dataset, groups[current_uid], torch.stack(pending_video[:n]), torch.stack(pending_text[:n]))
            torch.save(shard, osp.join(shards_dir, f"{current_uid}.pt.tmp"))
            os.replace(osp.join(shards_dir, f"{current_uid}.pt.tmp"), osp.join(shards_dir, f"{current_uid}.pt"))
            pending_video, pending_text = pending_video[n:], pending_text[n:]
            current_uid = next(remaining, None)

    # Merge the shards into features.pt once every video is done (possibly by other jobs)
    missing = [uid for uid in shard_uids if not osp.exists(osp.join(shards_dir, f"{uid}.pt"))]
    if missing:
        logger.info("%d / %d videos done; features.pt is written once the remaining %d are.", len(shard_uids) - len(missing), len(shard_uids), len(missing))
        return

    logger.info("Merging %d shards...", len(shard_uids))
    features = {}
    for uid in tqdm(shard_uids, desc="Merging shards"):
        shard = torch.load(osp.join(shards_dir, f"{uid}.pt"), weights_only=True)
        if is_mcq:
            features.update(shard)
        else:
            features[uid] = shard
    torch.save(features, osp.join(cfg.output_dir, "features.pt"))
    logger.info("Saved the features of %d %s to %s", len(features), "questions" if is_mcq else "videos", osp.join(cfg.output_dir, "features.pt"))


if __name__ == "__main__":
    main()

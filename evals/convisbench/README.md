# ConViS-Bench evaluation

For each annotated video pair and each concept (main action, main subjects, main objects, location,
order of actions), the similarity a model assigns to the two videos is correlated (Spearman ρ, Kendall τ)
with the human similarity scores.

| `--model_path` | Table 1 row | Similarity between the two videos |
| --- | --- | --- |
| `PE-Core-L14-336` | Perception Encoder, CLIPScore | raw Perception Encoder embedding of one video vs. the text embedding of the other video's caption |
| `splice` | SpLiCE, CLIPScore | the same, on the SpLiCE reconstructed embedding |
| `grace` | GRACE, CLIPScore | the same, on the GRACE reconstructed embedding |
| `splice_weights` | SpLiCE, Weights (L1) | negative L1 distance between the SpLiCE weights |
| `grace_weights` | GRACE, Weights (JSD) | negative Jensen–Shannon distance between the GRACE reconstruction weights |

## Running it

From the repository root, with the ConViS-Bench videos in `$GRACE_DATA/convisbench/`
(`python -m scripts.download_data`), in three steps:

```bash
python -m evals.convisbench.extract_features --model_path grace_weights \
  --ckpt checkpoints/video/grace_full_vocab.pth
python -m evals.convisbench.compute_similarities --model_path grace_weights
python -m evals.convisbench.compute_conditioned_correlation --models grace_weights
```

1. `extract_features` saves one file per video (16 uniformly sampled frames) to
   `visual_features/<model_path>/`: the video embedding, or the concept weights for `*_weights`
   (~3 min). Pass `--ckpt` for `grace` / `grace_weights`; for `splice` / `splice_weights`, pass
   `--splice-centering True --splice-l1-penalty 0.15` (the vocabulary defaults to the mined top-25
   EgoClip verbs and nouns, the visual mean is `data/means/egoclip_PE-Core-L14-336.pt`). The directory doesn't depend on
   `--ckpt`: delete it when switching checkpoints.
2. `compute_similarities` saves the similarity of every pair to `computed_similarities/<model_path>.json`.
   The CLIPScore rows use the mPLUG-Owl3 captions in `captions/mPLUG-Owl3-7B-240728/`.
3. `compute_conditioned_correlation` prints the table; `--models` takes one or more rows (default: all
   five, from the similarities shipped in `computed_similarities/`).

## Contents

- `extract_features.py`, `compute_similarities.py`, `compute_conditioned_correlation.py`: the three steps.
- `video_dataset.py`: decodes the videos of the annotated pairs.
- `captions/mPLUG-Owl3-7B-240728/`: mPLUG-Owl3 captions of the videos (from ConViS-Bench).
- `human_annotations/vicos.user_inputs.json`: the human similarity annotations (from ConViS-Bench).
- `computed_similarities/`: the pair similarities of the five rows.

<div align="center">

# GRACE: A General Concept-based Decomposition for Vision–Language Embeddings

[Simone Alberto Peirone](https://scholar.google.com/citations?user=K0efPssAAAAJ)<sup>1</sup> • [Ortal Senouf](https://scholar.google.com/citations?user=zv_5eUsAAAAJ)<sup>2</sup> • [Francesca Pistilli](https://scholar.google.com/citations?user=7MJdvzYAAAAJ)<sup>1</sup> • [Giuseppe Averta](https://scholar.google.com/citations?user=i4rm0tYAAAAJ)<sup>1</sup> • [Pascal Frossard](https://scholar.google.com/citations?user=-Ve9sJ0AAAAJ)<sup>2</sup>

<sup>1</sup>Politecnico di Torino • <sup>2</sup>EPFL

✨&nbsp;<b>NeurIPS 2026</b>&nbsp;✨

</div>

<div align="center">
  <a href='https://arxiv.org/abs/xxxx.xxxxx' style="margin: 10px"><img src='https://img.shields.io/badge/Paper-Arxiv:xxxx.xxxxx-red'></a>
  <a href='https://sapeirone.github.io/grace/' style="margin: 10px"><img src='https://img.shields.io/badge/Project-Page-Green'></a>
</div>

<br>

<div align="center">
<figure>
<img style="max-height: 480px" src="docs/teaser.gif" alt="The GRACE framework"/>
</figure>
</div>


## 📝 Abstract

Vision-Language Models are designed to capture the compositional structure of the world through multi-modal alignment; however, it remains unclear whether their representations truly align with the way a human would naturally describe the same visual input. Prior works attempt to decompose visual embeddings into their concept-level contributions, but their instance-level optimization is inherently local and does not capture the model's global semantic structure, which is essential for compositional generalization across different inputs, tasks and domains. We introduce GRACE, a method that learns this concept-based structure at the feature space level, producing semantic and general concept decompositions. To achieve this, we propose a novel training objective that preserves the geometric relations of different samples in the concept space, enabling a better understanding of how concepts compose and relate to each other. We validate GRACE on several image and video benchmarks, showing that its concept decomposition is aligned with human perception and achieves faithful decompositions across different tasks and domains. With minimal overhead, GRACE makes existing Vision-Language Models more interpretable, while preserving most of their original performance.

## 🔎 Getting started

This repository contains the code and configs, and links to the released checkpoints, to reproduce the main experiments in the paper, on both video (**EgoMCQ**, **Charades-Ego** and **ConViS-Bench**) and image benchmarks (**ImageNet**, **Flowers102**, **Food101**, **COCO**, **Flickr30k**, **ADE20k**).

### 🏗️ Environment setup

This repository was tested with Python 3.11 and CUDA 12.4 wheels, with [uv](https://docs.astral.sh/uv/).
To create a virtual environment, run the following lines:
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

uv also reads the `[tool.uv]` settings in `pyproject.toml` (PyTorch index, a setuptools pin needed to build openai/CLIP, a trimmed dependency list for Perception Encoder), so run it from the repository root.
All commands below assume that this environment is activated and that the repository root is the working directory.

### 📦 Data

All datasets live under a shared data root.
You can specify the path to the data root with the `GRACE_DATA` environment variable, or use the `grace_data/` path in the repository root.

We provide a `scripts/download_data.py` script to download all the publicly available datasets, which include Charades-Ego, ConViS-Bench, COCO (val), ADE20k, Flowers102, Food101 and the EgoClip / EgoMCQ annotations. 
The remaining data (ImageNet, Flickr30k and the Ego4D metadata) need a license or a login. 
Please download them separately into the data directory, then run the `scripts/check_data.py` script to check that everything is set up correctly.

```bash
export GRACE_DATA=/path/to/grace_data
python -m scripts.download_data   # public datasets
python -m scripts.check_data      # check what's there, what's missing and how to get it
```

See [DATA.md](DATA.md) for the expected layout, sizes, licenses and step-by-step instructions
for each dataset and for the precomputed features.

## 🐘 Model Zoo

<!-- TODO: replace the placeholder link below with the checkpoints link -->
Download the checkpoints from [this link](https://TODO-checkpoints-link) and place them in the `checkpoints/`
directory of the repository: the video checkpoints (`.pth`) in `checkpoints/video/`, the image ones (`.pt`)
in `checkpoints/image/`. All commands refer to them by these paths, e.g. `checkpoints/video/grace.pth`.

```
checkpoints/
├── video/   # grace.pth, grace_full_vocab.pth, ...
└── image/   # grace.pt, grace_tipsv2_l14_laion.pt, ...
```

The main ones:

| Checkpoint                        | Backbone        | Vocabulary (size)                | Key results                                                       |
|-----------------------------------|-----------------|----------------------------------|-------------------------------------------------------------------|
| `video/grace.pth`                 | PE-Core-L14-336 | EgoClip verbs + nouns (700)      | EgoMCQ Acc. 35.43 / 81.35 (intra / inter), Charades-Ego mAP 25.99 |
| `video/grace_full_vocab.pth`      | PE-Core-L14-336 | Top-25 verbs + nouns (2139)      | ConViS-Bench Corr. 56.3 / 40.1 (ρ / τ)                            |
| `image/grace.pt`                  | CLIP ViT-B/32   | ImageNet caption lemmas (15,162) | ImageNet Acc. 48.04, COCO R@1 41.80 / 33.56 (I→T / T→I)           |
| `image/grace_tipsv2_l14_laion.pt` | TIPSv2-L14      | LAION bigrams (15,000)           | ADE20k AUPRC 0.482                                                |

All the concept vocabularies are in `data/concepts_vocs/{video,image}/`.

## ⚡ Quickstart

Decompose an image into concepts with the released CLIP checkpoint:

```bash
python -m evals.quickstart path/to/image.jpg --top-k 10
```

```python
import torch
from PIL import Image

from models.backbones.clip import CLIPBackbone
from models.checkpoint import load_checkpoint

backbone = CLIPBackbone().cuda().eval()
grace = load_checkpoint("checkpoints/image/grace.pt", "cuda")

image = backbone.preprocess(Image.open("image.jpg").convert("RGB"))
with torch.no_grad():
    features = backbone.encode_image(image[None].cuda())
    reconstructed, weights = grace.encode_image(features, return_weights=True)
top_weights, top_indices = weights[0].topk(10)
print([(grace.text_concepts[i], round(w, 3)) for w, i in zip(top_weights.tolist(), top_indices.tolist())])
```

`reconstructed` is the concept-based embedding, a drop-in replacement for `features`; `weights` are the
per-concept reconstruction weights.

## 📊 Evaluation

Run the following commands from the repository root. 

### EgoMCQ

Video-text retrieval from the precomputed EgoMCQ features:

```bash
# GRACE
python -m evals.egomcq --config-name grace/egomcq resume_from=checkpoints/video/grace.pth

# SpLiCE baseline
python -m evals.egomcq --config-name splice/egomcq
```

### Charades-Ego

Zero-shot action recognition:

```bash
# GRACE 
python -m evals.charades --config-name grace/charades resume_from=checkpoints/video/grace.pth

# SpLiCE
python -m evals.charades --config-name splice/charades
```

### ConViS-Bench

The model is asked to estimate the similarity between two videos and performance is measured in terms of correlation (Spearman ρ, Kendall τ) with human judgement.
The evaluation code is adapted from the original ConViS-Bench repository, and the GRACE model is trained with the larger mined vocabulary, `checkpoints/video/grace_full_vocab.pth`. 
This repository lets you reproduce the results with the baseline Perception Encoder model, SpLiCE and GRACE.

| `--model_path`    | Table 1 row                   | Similarity between the two videos                                                         |
|-------------------|-------------------------------|-------------------------------------------------------------------------------------------|
| `PE-Core-L14-336` | Perception Encoder, CLIPScore | One video's embedding vs. the text embedding of the other video's caption                 |
| `splice`          | SpLiCE, CLIPScore             | The same, with the SpLiCE reconstructed embedding                                         |
| `splice_weights`  | SpLiCE, Weights (L1)          | Negative L1 distance between the SpLiCE weights                                           |
| `grace`           | GRACE, CLIPScore              | The same, with the GRACE reconstructed embedding                                          |
| `grace_weights`   | GRACE, Weights (JSD)          | Negative Jensen–Shannon distance between the GRACE reconstruction weights                 |


The evaluation has three steps: (i) extract per-video features (~3 min, saved to `evals/convisbench/visual_features/<model_path>/`), (ii) compute the pair similarities (saved to `evals/convisbench/computed_similarities/<model_path>.json`), and (iii) compute the correlation with the human annotations:

```bash
python -m evals.convisbench.extract_features --model_path grace_weights --ckpt checkpoints/video/grace_full_vocab.pth
python -m evals.convisbench.compute_similarities --model_path grace_weights
python -m evals.convisbench.compute_conditioned_correlation --models grace_weights
```
See [evals/convisbench/README.md](evals/convisbench/README.md) for more details.

### Image classification

Zero-shot classification (Table 6): pick a dataset with `--config-name` (`grace/imagenet`,
`grace/flowers102`, `grace/food101`), or the training-free SpLiCE baseline with `splice/<dataset>`:

```bash
python -m evals.image_classification --config-name grace/imagenet resume_from=checkpoints/image/grace.pt
python -m evals.image_classification --config-name splice/imagenet
```

### Image-text retrieval

COCO and Flickr30k retrieval (Table 5): `--config-name grace/coco` or `grace/flickr30k` (`splice/<dataset>`
for SpLiCE):

```bash
python -m evals.image_retrieval --config-name grace/coco resume_from=checkpoints/image/grace.pt
python -m evals.image_retrieval --config-name splice/coco
```

> [!NOTE]
> The CLIP (Flickr30k) and SpLiCE rows of Tables 5 and 6 are reported from EXPLAIN-R. Running them with this repository gives slightly different values: CLIP on Flickr30k obtains 66.01 / 62.48 R@1 (I→T / T→I) instead of 67.17 / 63.60, SpLiCE is within 2 points on retrieval and differs by up to 5 points of faithfulness on Flowers102 and Food101.

### ImageNet class folds

Transfer under concept shift (Figure 4): a GRACE model trained on the 800 training classes of a fold, evaluated on
the validation images of its 200 held-out classes (classified among all 1000):

```bash
python -m evals.image_classification --config-name grace/imagenet \
  resume_from=checkpoints/image/grace_laion_fold0.pt \
  folds_path=data/folds/imagenet_class_folds_concept_disjoint.json fold_id=0
```

### ADE20k concept grounding

Patch-level concept grounding with the TIPSv2 backbone (Table 9, ~17 min), and the figure of the top
concepts' heatmaps (Figure 6):

```bash
# To compute the image segmentation results
python -m evals.ade20k --config-name grace/ade20k resume_from=checkpoints/image/grace_tipsv2_l14_laion.pt

# To generate a concepts heatmap
python -m evals.ade20k_visualize --config-name grace/ade20k resume_from=checkpoints/image/grace_tipsv2_l14_laion.pt vis_output=top_concepts.jpg
```

## 🚀 Training

Checkpoints are saved to `checkpoints/runs/<timestamp>_<name>/` in the same format as the released ones,
so the evaluation commands above load them with `resume_from=...`. 

### Video

To train GRACE on EgoClip:

```bash
python -m train --config-name grace/egomcq
```

### Image

To train GRACE on ImageNet:

```bash
python -m scripts.extract_imagenet_features
python -m train_image --config-name grace/imagenet_train
```

## 🙏 Acknowledgements

This work builds on [ConViS-Bench](https://benedettaliberatori.github.io/convisbench/), [SpLiCE](https://github.com/AI4LIFE-GROUP/SpLiCE),
[EgoVLP](https://github.com/showlab/EgoVLP) (EgoClip / EgoMCQ),
[Perception Encoder](https://github.com/facebookresearch/perception_models),
[CLIP](https://github.com/openai/CLIP) and [TIPSv2](https://huggingface.co/google/tipsv2-l14), and on
the Ego4D, Charades-Ego, ImageNet, COCO, Flickr30k, Flowers102, Food101 and ADE20k datasets.

## 📜 License

The code is released under the [MIT License](LICENSE). The ConViS-Bench evaluation code under `evals/convisbench/` comes from the ConViS-Bench repository.

## 📚 Cite us

If you use this code, please cite our NeurIPS 2026 paper:

```bibtex
@inproceedings{peirone2026grace,
  title     = {A General Concept-based Decomposition for Vision-Language Embeddings},
  author    = {Peirone, Simone Alberto and Senouf, Ortal and Pistilli, Francesca and Averta, Giuseppe and Frossard, Pascal},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

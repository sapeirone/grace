# Data

All datasets live under a shared data root.
You can specify the path to the data root with the `GRACE_DATA` environment variable, or use the `grace_data/` path in the repository root.

We provide a `scripts/download_data.py` script to download all the publicly available datasets, which includes Charades-Ego, ConViS-Bench, COCO (val), ADE20k, Flowers102, Food101 and the EgoClip / EgoMCQ annotations. 
The remaining data (ImageNet, Flickr30k and the Ego4D metadata) need a license or a login. 
Please download them separately in the data directory, then run the `scripts/check_data.py` script to check that everything is set up correctly.

```bash
export GRACE_DATA=/path/to/grace_data
python -m scripts.download_data   # public datasets
python -m scripts.check_data      # check what's there, what's missing and how to get it
```

`download_data` skips what's already in place and checks the md5 of the files that have a published one. 
Archives are deleted after extraction unless `--keep-archives` is given, so expect some extra temporary space while a dataset is being extracted. 
Both scripts take `--only <dataset> ...`.

## What you need

| In order to run | You need the following |
| --- | --- |
| **Video evals**: EgoMCQ | `ego4d-annotations`, `egomcq-features` |
| **Video evals**: Charades-Ego | `charades_ego` |
| **Video evals**: ConViS-Bench | `convisbench` |
| **Image evals**: classification | `imagenet-val`, `flowers102`, `food101` |
| **Image evals**: retrieval | `coco`, `flickr30k` |
| **Image evals**: ADE20k grounding | `ade20k` |
| **Video training** | `ego4d-annotations`, `ego4d-metadata`, `egoclip-features` (+ `egomcq-features` for the final eval) |
| **Image training** | `imagenet-train`, `imagenet-features` |

## Datasets

| Dataset | Source | How to download | Size |
| --- | --- | --- | --- |
| `ego4d-annotations` | EgoClip / EgoMCQ annotations released by [EgoVLP](https://github.com/showlab/EgoVLP) | `download_data` | 0.7 GB | 
| `ego4d-metadata` | `ego4d.json` from [Ego4D](https://ego4d-data.org/docs/start-here/) | manual, needs the Ego4D license | 70 MB | 
| `egomcq-features` | PE-Core-L14-336 features of the EgoMCQ clips | not distributed, see [Precomputed features](#precomputed-features) | 7.5 GB |
| `egoclip-features` | PE-Core-L14-336 features of the EgoClip clips | not distributed, see [Precomputed features](#precomputed-features) | 16 GB |
| `charades_ego` | [Charades-Ego](https://prior.allenai.org/projects/charades-ego) annotations and 480p videos (AI2) | `download_data` (slow server: several hours) | 11 GB |
| `convisbench` | [ConViS-Bench](https://huggingface.co/datasets/bliberatori/ConViS-Bench) video pairs and annotations (pinned revision) | `download_data` | 1.2 GB | 
| `coco` | [COCO](https://cocodataset.org) 2017 validation images and captions | `download_data` | 0.8 GB |
| `flickr30k` | [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) images and captions | manual request | 4.2 GB |
| `imagenet-val` | [ILSVRC2012](https://image-net.org/download.php) validation images | manual, account required | 6.3 GB |
| `imagenet-train` | ILSVRC2012 training images | manual, account required | 137 GB |
| `imagenet-features` | CLIP ViT-B/32 features of ImageNet train | `python -m scripts.extract_imagenet_features` (~1 h) | 2.6 GB |
| `flowers102` | [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), via torchvision | `download_data` | 0.3 GB |
| `food101` | [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), via torchvision | `download_data` | 4.8 GB |
| `ade20k` | [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/) SceneParsing (`ADEChallengeData2016`) | `download_data` | 0.9 GB |

None of these datasets is redistributed by this repository. The released checkpoints are trained on
features of EgoClip (video) and ImageNet (image), so their use is also subject to those datasets' terms.

## Layout

```
$GRACE_DATA/
├── ego4d/
│   ├── annotations/
│   │   ├── egoclip.csv                         # EgoVLP release
│   │   ├── egomcq.json                         # EgoVLP release
│   │   └── ego4d.json                          # Ego4D metadata (scenario tags)
│   └── features/
│       ├── EgoClip_PE-Core-L14-336/features.pt
│       └── EgoMCQ_PE-Core-L14-336/features.pt
├── charades_ego/
│   ├── annotations/                            # CharadesEgo_v1_test_only1st.csv, Charades_v1_classes.txt, ...
│   └── videos/                                 # 7,860 .mp4
├── convisbench/
│   ├── ConVIS.jsonl
│   └── videos/                                 # 543 .mp4
├── coco/
│   ├── annotations/captions_val2017.json
│   └── val2017/                                # 5,000 .jpg
├── flickr30k/
│   ├── flickr30k_all.tsv                       # 158,915 captions, `<image>#<i>\t<caption>` per line
│   └── Images/                                 # 31,783 .jpg
├── imagenet/
│   ├── val/<wnid>/                             # 50,000 .JPEG, one folder per class
│   ├── train/<wnid>/                           # 1,281,167 .JPEG (training only)
│   └── train_features_clip_ViT-B-32.pt         # training only
├── flowers-102/                                # imagelabels.mat, setid.mat, jpg/ (8,189 images)
├── food-101/                                   # meta/, images/ (101,000 images)
└── ade20k/ADEChallengeData2016/                # images/ and annotations/ (20,210 train, 2,000 val)
```

`check_data` verifies these files and the image and video counts (`--quick` only checks that they
exist).

## Manual downloads

- **Ego4D metadata**: accept the license at [ego4ddataset.com](https://ego4ddataset.com) and install the
  [Ego4D CLI](https://ego4d-data.org/docs/start-here/). It downloads `ego4d.json` along with any
  dataset; copy it to `ego4d/annotations/ego4d.json`. Only video training needs it (Ego4D scenario labels,
  used as tags by the consistency loss).
- **Flickr30k**: request the dataset on its [page](https://shannon.cs.illinois.edu/DenotationGraph/). Put
  the 31,783 images in `flickr30k/Images/` and the captions in `flickr30k/flickr30k_all.tsv`, in the
  format of the official `results_20130124.token` (`<image>#<i>\t<caption>`, one line per caption).
- **ImageNet**: download ILSVRC2012 from [image-net.org](https://image-net.org/download.php) and put the
  images in `imagenet/{val,train}/<wnid>/`, one folder per class (the torchvision `ImageFolder` layout).
  The evals only need `val`; `train` is only needed to extract the training features.

## Precomputed features

The video models are trained and evaluated on precomputed Perception Encoder features, and image
training on precomputed CLIP features. They aren't distributed (they're derived from Ego4D and
ImageNet), so they must be extracted locally.

- **ImageNet** (`imagenet/train_features_clip_ViT-B-32.pt`): `python -m scripts.extract_imagenet_features`
  (CLIP ViT-B/32 with the evaluation transform, ~1 h on one GPU). A dict mapping each image's index in
  `ImageFolder(imagenet/train)` order to its `(512,)` feature; the labels come from the folder layout.
- **EgoClip / EgoMCQ** (`ego4d/features/{EgoClip,EgoMCQ}_PE-Core-L14-336/features.pt`): PE-Core-L14-336
  video and text features of the EgoClip clips and EgoMCQ candidates (16 frames per clip), extracted from
  the Ego4D videos (`<video_uid>.mp4`, from the Ego4D CLI) with:

  ```bash
  python -m scripts.extract_ego4d_features dataset=egomcq dataset.videos_root=/path/to/ego4d/videos \
    output_dir=$GRACE_DATA/ego4d/features/EgoMCQ_PE-Core-L14-336
  python -m scripts.extract_ego4d_features dataset=egoclip dataset.videos_root=/path/to/ego4d/videos \
    output_dir=$GRACE_DATA/ego4d/features/EgoClip_PE-Core-L14-336
  ```

  Each video's features are saved to `<output_dir>/shards/` as soon as it's done, so an interrupted run
  resumes where it stopped; `num_slices=<n> slice=<k>` splits the videos across parallel jobs. Once all
  shards are there, they're merged into `features.pt` (EgoClip's merge needs ~16 GB of RAM); the shards
  can then be deleted. Videos missing from `videos_root` are skipped. 

## Files in the repository

Small derived files are versioned under `data/`, so the evals don't need to recompute them:

| Path | Content |
| --- | --- |
| `data/concepts_vocs/video/` | Video concept vocabularies: EgoClip verbs + nouns (700), mined top-25% verbs + nouns (2139), GPT-5.4 (414) |
| `data/concepts_vocs/image/` | Image concept vocabularies: ImageNet caption lemmas (15,162), LAION bigrams (15,000, from SpLiCE) |
| `data/means/` | Visual means used to center SpLiCE (EgoClip for video; the evaluated split for each image dataset) |
| `data/folds/` | ImageNet class folds with minimal concept overlap (Figure 4) |
| `data/classnames/` | Class names for zero-shot classification on ImageNet and Flowers102 (Food101's come from torchvision) |

The ConViS-Bench mPLUG-Owl3 captions and human annotations ship with its evaluation code, under
`evals/convisbench/`.

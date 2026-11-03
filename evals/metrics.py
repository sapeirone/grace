"""Shared metrics for the image classification/retrieval and Charades eval entrypoints."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CLASSNAMES_ROOT = Path(__file__).resolve().parent.parent / "data" / "classnames"


def get_classes(dataset_name: str, dataset=None) -> list[str]:
    """Get the human-readable class names for a classification dataset, in the same order
    as its integer labels (i.e. `classes[label]` is the prompt-ready class name).

    Parameters
    ----------
    dataset_name : str
        One of "imagenet", "flowers102", "food101".
    dataset : optional
        The instantiated dataset, used for "food101" (whose class names come from
        torchvision's own `.classes` attribute rather than a bundled text file).

    Returns
    -------
    list[str]
        Class names, `classes[i]` being the name of class `i`.
    """
    if dataset_name == "imagenet":
        with open(CLASSNAMES_ROOT / "imagenet.txt", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]

    if dataset_name == "flowers102":
        with open(CLASSNAMES_ROOT / "flowers102.txt", encoding="utf-8") as f:
            return [line.strip("' \n") for line in f.readlines()]

    if dataset_name == "food101":
        if dataset is None:
            raise ValueError("food101 class names require the instantiated dataset (uses its own `.classes`).")
        return dataset.classes

    raise ValueError(f"Unknown dataset: {dataset_name!r} for classification evaluator")


@torch.inference_mode()
def find_closest(embeddings: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
    """Zero-shot classification: assign each embedding to its closest label embedding."""
    return torch.argmax(embeddings @ label_embeddings.T, dim=-1)


def compute_retrieval(a2b_sims: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute R@1/5/10 retrieval metrics from a similarity matrix.

    Parameters
    ----------
    a2b_sims : np.ndarray
        Similarity matrix, shape (num_datapoints, num_datapoints).
    targets : np.ndarray
        Index of the correct match for each row of `a2b_sims`.

    Returns
    -------
    dict[str, float]
        R@1/5/10, as percentages.
    """
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    assert len(targets) == npts, "Number of targets must be the same as the number of embeddings."
    for index, target in enumerate(targets):
        inds = np.argsort(a2b_sims[index])[::-1]
        ranks[index] = np.where(inds == target)[0][0]

    return {
        "r1": 100.0 * len(np.where(ranks < 1)[0]) / len(ranks),
        "r5": 100.0 * len(np.where(ranks < 5)[0]) / len(ranks),
        "r10": 100.0 * len(np.where(ranks < 10)[0]) / len(ranks),
    }


def compute_retrieval_symmetric(similarity: np.ndarray, labels_i2t: np.ndarray, labels_t2i: np.ndarray) -> dict[tuple[str, str], float]:
    """Compute R@1/5/10 in both the image->text and text->image directions."""
    return {("i2t", metric): value for metric, value in compute_retrieval(similarity, labels_i2t).items()} | {
        ("t2i", metric): value for metric, value in compute_retrieval(similarity.T, labels_t2i).items()
    }


@torch.inference_mode()
def retrieval_metrics(student_sim: torch.Tensor, target_sim: torch.Tensor, index: Sequence[str]) -> pd.DataFrame:
    """Compute retrieval "performance" (vs. ground truth) and "faithfulness" (vs. the
    underlying CLIP model's own predictions) metrics for one similarity matrix."""
    target_i2t_preds = target_sim.argmax(dim=1).detach().cpu().numpy()
    target_t2i_preds = target_sim.argmax(dim=0).detach().cpu().numpy()
    labels_i2t = np.arange(target_sim.shape[0])
    labels_t2i = np.arange(target_sim.shape[1])

    student_sim_np = student_sim.detach().cpu().numpy()

    student_performance = compute_retrieval_symmetric(student_sim_np, labels_i2t, labels_t2i)
    student_faithfulness = compute_retrieval_symmetric(student_sim_np, target_i2t_preds, target_t2i_preds)

    columns = pd.MultiIndex.from_tuples(student_performance.keys())
    return pd.DataFrame([student_performance.values(), student_faithfulness.values()], index=index, columns=columns)


class RunningRetrievalStats:
    """Accumulates retrieval performance/faithfulness metrics across batches."""

    def __init__(self):
        self.stats = []

    def update(self, student_sim: torch.Tensor, target_sim: torch.Tensor) -> pd.DataFrame:
        """Compute this batch's retrieval metrics (in-batch retrieval: `student_sim`/`target_sim`
        must be square, one row/column per sample in the batch) and add them to the running total.
        """
        assert student_sim.shape == target_sim.shape, "student and target similarity matrices must have the same shape"
        assert student_sim.ndim == 2, "student and target similarity matrices must be 2D"
        assert student_sim.shape[0] == student_sim.shape[1], "student and target similarity matrices must be square"

        student_metrics = retrieval_metrics(student_sim, target_sim, ["Ours performance", "Ours faithfulness"])
        target_metrics = retrieval_metrics(target_sim, target_sim, ["Target performance", "Target faithfulness"]).iloc[:-1]

        df = pd.concat([student_metrics, target_metrics], axis=0)
        self.stats.append(df)
        return df

    def summary(self) -> pd.DataFrame:
        """Average the per-batch metrics collected so far."""
        return sum(self.stats) / len(self.stats)


def charades_map(scores: np.ndarray, gt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Multi-label mean average precision, as in the Charades evaluation code (an approximation
    of the official Matlab script).

    Parameters
    ----------
    scores : np.ndarray
        `(num_samples, num_classes)` prediction scores
    gt : np.ndarray
        `(num_samples, num_classes)` multi-hot ground truth

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        mAP, the per-class APs weighted by class frequency, and the per-class APs (NaN for classes
        without positives)
    """
    scores = scores.copy()
    # Samples without any positive label never count as a true positive.
    scores[gt.sum(axis=1) == 0, :] = -np.inf

    aps = []
    for c in range(scores.shape[1]):
        tp = gt[:, c][np.argsort(-scores[:, c])] == 1
        n_pos = tp.sum()
        if n_pos < 0.1:
            aps.append(float("nan"))
            continue
        precision = np.cumsum(tp) / (np.cumsum(~tp) + np.cumsum(tp)).astype(float)
        aps.append(precision[tp].sum() / float(n_pos))
    aps = np.array(aps)
    weighted_aps = aps * gt.sum(axis=0) / float(gt.sum())
    return float(np.mean(aps)), weighted_aps, aps

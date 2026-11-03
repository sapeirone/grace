"""
ConViS-Bench, step 3: correlate the pair similarities of step 2 with the human similarity annotations,
per concept (Spearman ρ / Kendall τ, as in Table 1), and print the table.

Usage (from the repository root):
    python -m evals.convisbench.compute_conditioned_correlation --models grace_weights
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr
from tabulate import tabulate

# Default paths are relative to these files, whatever the CWD
CONVIS_DIR = Path(__file__).resolve().parent

# `--model_path` of steps 1-2 -> row of Table 1
MODELS = {
    "PE-Core-L14-336": "Perception Encoder (CLIPScore)",
    "splice": "SpLiCE (CLIPScore)",
    "grace": "GRACE (CLIPScore)",
    "splice_weights": "SpLiCE (Weights, L1)",
    "grace_weights": "GRACE (Weights, JSD)",
}
# Annotated concept -> column title, in the order of Table 1
CONCEPTS = {
    "MainAction": "Main Action",
    "MainSubjects": "Main Subjects",
    "MainObjects": "Main Objects",
    "Location": "Location",
    "OrderOfActions": "Order of Actions",
}


def load_human_similarities(user_inputs_path: str) -> dict:
    """{concept: {(video1, video2): similarity averaged over the annotators}}."""
    with open(user_inputs_path, "r") as f:
        data = json.load(f)

    scores = defaultdict(lambda: defaultdict(list))
    for entry in data:
        video1, video2 = entry["video_pair_id"].split("-")
        inputs = entry.get("user_inputs", {})
        for concept in CONCEPTS:
            # The annotation files spell this one "OrderofActions"
            scores[concept][(video1, video2)].append(float(inputs.get("OrderofActions" if concept == "OrderOfActions" else concept)))
    return {concept: {pair: np.mean(values) for pair, values in pairs.items()} for concept, pairs in scores.items()}


def main(args):
    human = load_human_similarities(args.user_inputs)

    rows = []
    for model in args.models:
        with open(os.path.join(args.similarity_dir, f"{model}.json"), "r") as f:
            similarities = {(os.path.basename(e["video1"]), os.path.basename(e["video2"])): e["similarity"] for e in json.load(f)}

        results = {}
        for concept in CONCEPTS:
            pairs = [pair for pair in similarities if pair in human[concept]]
            predicted, annotated = [similarities[p] for p in pairs], [human[concept][p] for p in pairs]
            results[concept] = (
                round(spearmanr(predicted, annotated).correlation * 100, 2),
                round(kendalltau(predicted, annotated).correlation * 100, 2),
            )
        results["Average"] = tuple(round(np.mean([r[i] for r in results.values()]), 2) for i in range(2))
        rows.append([MODELS[model]] + [f"{value:.1f}" for rho_tau in results.values() for value in rho_tau])

    headers = ["Model"] + [f"{title} ({symbol})" for title in [*CONCEPTS.values(), "Average"] for symbol in ("ρ", "τ")]
    print("Correlation with the human annotations:")
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlate the ConViS-Bench pair similarities with the human annotations.")
    parser.add_argument(
        "--models", type=str, nargs="+", default=list(MODELS), choices=list(MODELS), help="Rows to report (default: all five)."
    )
    parser.add_argument("--similarity_dir", type=str, default=str(CONVIS_DIR / "computed_similarities"), help="Similarities of step 2.")
    parser.add_argument(
        "--user_inputs", type=str, default=str(CONVIS_DIR / "human_annotations" / "vicos.user_inputs.json"), help="Human annotations."
    )
    main(parser.parse_args())

"""
Shared setup of the command-line entrypoints (`train.py`, `train_image.py`, `evals/*.py`).
"""

import logging

BANNER = "A General Concept-based Decomposition for Vision–Language Embeddings"


def log_banner(logger: logging.Logger) -> None:
    """Log the paper title, framed, at the start of a run."""
    frame = "#" * (len(BANNER) + 4)
    for line in ("", frame, f"# {BANNER} #", frame, ""):
        logger.info(line)

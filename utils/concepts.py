import logging

import torch
from torch.nn import functional as F

from models.backbones.base import CLIP

logger = logging.getLogger(__name__)


def load_concepts(path: str, backbone: CLIP) -> tuple[list[str], torch.Tensor]:
    """Load a concept vocabulary and encode it with the backbone's text encoder.

    Parameters
    ----------
    path : str
        Vocabulary file: one concept per line, or a `pos,lemma` CSV table.
    backbone : CLIP
        Backbone whose text encoder embeds the concepts.

    Returns
    -------
    tuple[list[str], torch.Tensor]
        The concepts, and their embeddings `(num_concepts, features_size)`: normalized, centered on
        their mean and normalized again.
    """
    logger.info("")
    logger.info("Loading concepts from %s...", path)

    with open(path, encoding="utf-8") as f:
        lines = [line.lower().strip() for line in f if line.strip()]

    if path.endswith(".csv"):
        # A `pos,lemma` table (e.g. the ImageNet caption lemmas): drop the header and the lemmas that
        # repeat with a different part of speech, keeping the first occurrence.
        text_concepts = list(dict.fromkeys(line.split(",")[1] for line in lines[1:]))
    else:
        # One concept per line
        text_concepts = lines

    logger.info("Loaded %d concepts from %s", len(text_concepts), path)
    logger.info("")

    # Chunked so large vocabularies (e.g. 15k LAION bigrams) fit in GPU memory.
    concepts_embds = torch.cat([backbone.encode_text(text_concepts[i : i + 1024]) for i in range(0, len(text_concepts), 1024)])
    concepts_embds = F.normalize(concepts_embds, p=2, dim=1)
    concepts_embds = F.normalize(concepts_embds - concepts_embds.mean(0), p=2, dim=1)

    return text_concepts, concepts_embds

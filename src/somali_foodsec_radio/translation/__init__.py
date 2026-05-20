"""Translate Somali transcripts into English.

:mod:`~somali_foodsec_radio.translation.chunking` is pure and dependency-light, so it
is re-exported here. :mod:`~somali_foodsec_radio.translation.translate_hf` and
:mod:`~somali_foodsec_radio.translation.pipeline` pull in torch/transformers — import
those by their full module path when translating.
"""

from .chunking import (
    create_semantic_chunks,
    deduplicate_overlap,
    split_into_sentences,
)

__all__ = [
    "create_semantic_chunks",
    "deduplicate_overlap",
    "split_into_sentences",
]

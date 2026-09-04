"""Sentence splitting and token-aware chunking for translation.

Pure functions — no ML dependencies — so they import cheaply and are unit-tested
directly. :func:`create_semantic_chunks` needs a *tokenizer* exposing an
``encode(text, add_special_tokens=...)`` method (e.g. a HuggingFace tokenizer).
"""

from __future__ import annotations

import re
from typing import Any


def split_into_sentences(text: str) -> list[str]:
    """Split Somali *text* into sentence-like segments.

    Tries progressively coarser strategies: Somali connective words, full stops,
    common conjunctions, and finally fixed 50-word windows.
    """
    sentences = re.split(
        r"\s+(?=waxaa|waxa|marka|haddii|sida|taasi)", text, flags=re.IGNORECASE
    )
    if len(sentences) > 5:
        return [s.strip() for s in sentences if s.strip()]

    sentences = re.split(r"\.\s+", text)
    if len(sentences) > 3:
        return [s.strip() for s in sentences if s.strip()]

    sentences = re.split(r"\s+(ayaa|oo|iyo|ee)\s+", text, flags=re.IGNORECASE)
    result = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            result.append(sentences[i] + " " + sentences[i + 1])
        else:
            result.append(sentences[i])
    if len(result) > 2:
        return [s.strip() for s in result if s.strip()]

    words = text.split()
    chunk_size = 50
    sentences = [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]
    return [s.strip() for s in sentences if s.strip()]


def create_semantic_chunks(
    text: str,
    tokenizer: Any,
    max_tokens: int = 450,
    overlap_sentences: int = 2,
) -> list[tuple[str, int, int]]:
    """Split *text* into chunks that respect a token limit.

    Returns a list of ``(chunk_text, 0, 0)`` tuples — the trailing zeros are kept for
    backwards compatibility with callers that expected character offsets.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return [(text, 0, len(text))]

    print(f"  -> Found {len(sentences)} sentences/segments")

    chunks: list[tuple[str, int, int]] = []
    current_chunk: list[str] = []
    current_tokens = 0
    i = 0

    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=True))

        if sentence_tokens > max_tokens:
            print(
                f"  -> Sentence {i + 1} has {sentence_tokens} tokens, "
                f"splitting by words..."
            )
            words = sentence.split()
            word_chunk: list[str] = []
            word_tokens = 0
            for word in words:
                word_token_count = len(
                    tokenizer.encode(word + " ", add_special_tokens=False)
                )
                if word_tokens + word_token_count > max_tokens - 10:
                    if word_chunk:
                        chunks.append((" ".join(word_chunk), 0, 0))
                    word_chunk = [word]
                    word_tokens = word_token_count
                else:
                    word_chunk.append(word)
                    word_tokens += word_token_count
            if word_chunk:
                chunks.append((" ".join(word_chunk), 0, 0))
            i += 1
            continue

        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append((" ".join(current_chunk), 0, 0))
            overlap_start = max(0, len(current_chunk) - overlap_sentences)
            current_chunk = current_chunk[overlap_start:]
            current_tokens = sum(
                len(tokenizer.encode(s, add_special_tokens=True)) for s in current_chunk
            )

        current_chunk.append(sentence)
        current_tokens += sentence_tokens
        i += 1

    if current_chunk:
        chunks.append((" ".join(current_chunk), 0, 0))

    return chunks


def deduplicate_overlap(chunks: list[str], overlap_threshold: int = 5) -> str:
    """Merge translated *chunks*, detecting and removing overlapping word runs."""
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_chunk = result[-1]
        curr_chunk = chunks[i]
        prev_words = prev_chunk.split()
        curr_words = curr_chunk.split()
        max_overlap = min(len(prev_words), len(curr_words), 15)
        overlap_found = 0

        for overlap_size in range(max_overlap, overlap_threshold - 1, -1):
            prev_end = " ".join(prev_words[-overlap_size:])
            curr_start = " ".join(curr_words[:overlap_size])
            if prev_end.lower() == curr_start.lower():
                overlap_found = overlap_size
                break

        if overlap_found > 0:
            result.append(" ".join(curr_words[overlap_found:]))
        else:
            result.append(curr_chunk)

    return " ".join(result)

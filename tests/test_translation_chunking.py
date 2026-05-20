"""Tests for the pure translation chunking helpers."""

from somali_foodsec_radio.translation.chunking import (
    create_semantic_chunks,
    deduplicate_overlap,
    split_into_sentences,
)


class FakeTokenizer:
    """Minimal tokenizer stand-in: one token per whitespace-separated word."""

    def encode(self, text, add_special_tokens=True):  # noqa: D102, ARG002
        return text.split()


class TestSplitIntoSentences:
    def test_splits_on_somali_connectives(self):
        text = "a waxaa b waxaa c waxaa d waxaa e waxaa f waxaa g"
        result = split_into_sentences(text)
        assert len(result) == 7
        assert result[0] == "a"
        assert result[1] == "waxaa b"

    def test_falls_back_to_full_stops(self):
        result = split_into_sentences("One thing. Two thing. Three thing. Four thing.")
        assert len(result) == 4

    def test_falls_back_to_50_word_windows(self):
        words = " ".join(["word"] * 120)
        result = split_into_sentences(words)
        assert len(result) == 3  # 50 + 50 + 20


class TestCreateSemanticChunks:
    def test_short_text_is_one_chunk(self):
        chunks = create_semantic_chunks("hello world", FakeTokenizer(), max_tokens=450)
        assert len(chunks) == 1
        assert chunks[0][0] == "hello world"

    def test_long_text_splits_and_respects_token_limit(self):
        text = "alpha beta. gamma delta. epsilon zeta. eta theta."
        tok = FakeTokenizer()
        chunks = create_semantic_chunks(
            text, tok, max_tokens=5, overlap_sentences=1
        )
        assert len(chunks) > 1
        for chunk_text, _, _ in chunks:
            assert len(tok.encode(chunk_text)) <= 5

    def test_oversized_sentence_is_word_split(self):
        # A single 60-word "sentence" exceeds the limit and is split by words.
        text = " ".join(["w"] * 60)
        chunks = create_semantic_chunks(text, FakeTokenizer(), max_tokens=20)
        assert len(chunks) > 1


class TestDeduplicateOverlap:
    def test_empty_list_returns_empty_string(self):
        assert deduplicate_overlap([]) == ""

    def test_single_chunk_unchanged(self):
        assert deduplicate_overlap(["only one chunk"]) == "only one chunk"

    def test_no_overlap_is_concatenated(self):
        assert deduplicate_overlap(["a b c", "d e f"]) == "a b c d e f"

    def test_overlapping_tail_and_head_merged_once(self):
        merged = deduplicate_overlap(
            ["a b c brown fox jumps", "brown fox jumps d e f"],
            overlap_threshold=3,
        )
        assert merged == "a b c brown fox jumps d e f"

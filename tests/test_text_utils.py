"""Tests for the shared text-normalisation helper."""

from somali_foodsec_radio.text_utils import clean_text


def test_lowercases():
    assert clean_text("HELLO World") == "hello world"


def test_strips_accents():
    assert clean_text("Cádaado") == "cadaado"


def test_removes_punctuation():
    assert clean_text("Hello, World!") == "hello world"


def test_trims_whitespace():
    assert clean_text("   spaced out   ") == "spaced out"


def test_combined():
    assert clean_text("  Beled-Xaawo, Gedo!  ") == "beledxaawo gedo"

"""Tests for the pure theme-selection helper (no BERTopic model needed)."""

from somali_foodsec_radio.topics.bertopic_model import select_themes_per_doc

THEME_MAP = {
    0: "Rainfall",
    1: "Crop Failure",
    2: "Livestock Health",
    3: "Higher Food Prices",
    4: "Humanitarian Aid",
}


def test_selects_themes_above_threshold():
    probs = [[0.5, 0.02, 0.0, 0.2, 0.0]]
    assert select_themes_per_doc(probs, THEME_MAP, prob_threshold=0.1) == [
        ["Rainfall", "Higher Food Prices"]
    ]


def test_doc_below_threshold_gets_other():
    probs = [[0.05, 0.0, 0.0, 0.0, 0.0]]
    assert select_themes_per_doc(probs, THEME_MAP, prob_threshold=0.1) == [["Other"]]


def test_topic_id_absent_from_map_is_skipped():
    # Topic 5 clears the threshold but is not in THEME_MAP -> ignored.
    probs = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.9]]
    assert select_themes_per_doc(probs, THEME_MAP, prob_threshold=0.1) == [["Other"]]


def test_handles_multiple_documents():
    probs = [[0.5, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.3, 0.0, 0.0]]
    assert select_themes_per_doc(probs, THEME_MAP, prob_threshold=0.1) == [
        ["Rainfall"],
        ["Livestock Health"],
    ]

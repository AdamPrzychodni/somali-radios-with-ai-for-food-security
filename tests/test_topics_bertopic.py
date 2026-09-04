"""Tests for the pure theme helpers (no BERTopic model needed)."""

from somali_foodsec_radio.topics.bertopic_model import (
    select_themes_per_doc,
    topic_map_mismatches,
)

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


class TestTopicMapMismatches:
    """BERTopic numbers topics by cluster size, so a rerun can silently relabel them.

    The guard is what turns that into a loud failure instead of wrong themes.
    """

    THEME_KEYWORDS = {
        "Rainfall": ["rain", "rains", "rainfall"],
        "Crop Failure": ["crop", "harvest", "farm"],
    }

    def test_matching_topics_report_nothing(self):
        topic_words = {0: ["rain", "season", "water"], 1: ["harvest", "crop", "maize"]}
        assert (
            topic_map_mismatches(
                topic_words, {0: "Rainfall", 1: "Crop Failure"}, self.THEME_KEYWORDS
            )
            == {}
        )

    def test_swapped_ids_are_caught(self):
        topic_words = {0: ["harvest", "crop"], 1: ["rain", "season"]}
        assert topic_map_mismatches(
            topic_words, {0: "Rainfall", 1: "Crop Failure"}, self.THEME_KEYWORDS
        ) == {0: "Rainfall", 1: "Crop Failure"}

    def test_missing_topic_is_a_mismatch(self):
        assert topic_map_mismatches({}, {0: "Rainfall"}, self.THEME_KEYWORDS) == {
            0: "Rainfall"
        }

    def test_theme_without_configured_keywords_is_skipped(self):
        assert topic_map_mismatches({0: ["anything"]}, {0: "Unmapped"}, {}) == {}

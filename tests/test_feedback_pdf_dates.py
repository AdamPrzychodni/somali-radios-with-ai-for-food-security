"""Tests for the feedback-PDF date helpers."""

from somali_foodsec_radio.feedback.pdf_extract import clean_date, is_valid_date


class TestIsValidDate:
    def test_accepts_dash_date(self):
        assert is_valid_date("01-08-2022")

    def test_accepts_slash_date(self):
        assert is_valid_date("1/8/2022")

    def test_rejects_plain_text(self):
        assert not is_valid_date("hello")

    def test_rejects_year_only(self):
        assert not is_valid_date("2022")

    def test_rejects_two_digit_year(self):
        assert not is_valid_date("01-08-22")


class TestCleanDate:
    def test_reformats_dash_date(self):
        assert clean_date("01-08-2022") == "2022-08-01"

    def test_reformats_slash_date(self):
        assert clean_date("1/8/2022") == "2022-08-01"

    def test_returns_input_when_unparseable(self):
        assert clean_date("not a date") == "not a date"

    def test_returns_input_when_incomplete(self):
        assert clean_date("01-08") == "01-08"

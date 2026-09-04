"""Tests for the SoundCloud URL helpers (pure functions, no network)."""

from datetime import datetime

from somali_foodsec_radio.collection.urls import (
    extract_date_from_url,
    extract_username_from_url,
    generate_url_patterns,
    generate_urls_for_range,
    validate_soundcloud_url,
)


class TestValidateSoundcloudUrl:
    def test_accepts_track_url(self):
        assert validate_soundcloud_url(
            "https://soundcloud.com/radio-ergo/idaacadda-01-jul-2024"
        )

    def test_accepts_www_and_http(self):
        assert validate_soundcloud_url(
            "http://www.soundcloud.com/radio-ergo/some-track"
        )

    def test_rejects_profile_only_url(self):
        assert not validate_soundcloud_url("https://soundcloud.com/radio-ergo")

    def test_rejects_non_soundcloud(self):
        assert not validate_soundcloud_url("https://example.com/radio-ergo/track")


class TestExtractDateFromUrl:
    def test_parses_dd_mon_yyyy(self):
        assert extract_date_from_url(
            "https://soundcloud.com/radio-ergo/idaacadda-01-jul-2024"
        ) == datetime(2024, 7, 1)

    def test_parses_full_month_name(self):
        assert extract_date_from_url(
            "https://soundcloud.com/radio-ergo/idaacadda-9-march-2025"
        ) == datetime(2025, 3, 9)

    def test_parses_iso_date(self):
        # Regression: the original unpacked YYYY-MM-DD groups in the wrong order.
        assert extract_date_from_url(
            "https://soundcloud.com/radio-ergo/2024-07-01"
        ) == datetime(2024, 7, 1)

    def test_returns_none_without_date(self):
        assert (
            extract_date_from_url("https://soundcloud.com/radio-ergo/welcome-show")
            is None
        )

    def test_returns_none_for_invalid_month(self):
        assert (
            extract_date_from_url(
                "https://soundcloud.com/radio-ergo/idaacadda-01-foo-2024"
            )
            is None
        )


class TestExtractUsername:
    def test_extracts_username(self):
        assert (
            extract_username_from_url("https://soundcloud.com/radio-ergo")
            == "radio-ergo"
        )

    def test_handles_trailing_slash(self):
        assert (
            extract_username_from_url("https://soundcloud.com/radio-ergo/")
            == "radio-ergo"
        )


class TestGenerateUrlPatterns:
    def test_first_pattern_is_zero_padded_idaacadda(self):
        patterns = generate_url_patterns("radio-ergo", datetime(2024, 7, 1))
        assert patterns[0] == "https://soundcloud.com/radio-ergo/idaacadda-01-jul-2024"

    def test_generates_six_candidates(self):
        assert len(generate_url_patterns("radio-ergo", datetime(2024, 7, 1))) == 6


class TestGenerateUrlsForRange:
    def test_one_url_per_day_inclusive(self):
        pairs = generate_urls_for_range(
            "https://soundcloud.com/radio-ergo",
            datetime(2022, 1, 1),
            datetime(2022, 1, 3),
        )
        assert len(pairs) == 3
        assert pairs[0] == (
            datetime(2022, 1, 1),
            "https://soundcloud.com/radio-ergo/idaacadda-01-jan-2022",
        )
        assert pairs[-1][1].endswith("idaacadda-03-jan-2022")

    def test_strips_trailing_slash_and_uses_slug(self):
        pairs = generate_urls_for_range(
            "https://soundcloud.com/radio-ergo/",
            datetime(2022, 1, 1),
            datetime(2022, 1, 1),
            slug="show",
        )
        assert pairs[0][1] == "https://soundcloud.com/radio-ergo/show-01-jan-2022"

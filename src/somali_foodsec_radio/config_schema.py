"""Schema for ``config/config.yaml``.

Validated on load so a typo fails immediately with a readable message, instead of
surfacing as a ``KeyError`` twenty minutes into a GPU run — or, worse, as a silently
ignored setting. Every section forbids unknown keys: a config value nobody reads looks
like it should do something, which is how the ``feedback:`` section came to be dead.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectSection(_Section):
    name: str
    seed: int = 42


class PathsSection(_Section):
    # `Path` as well as `str` so the schema also validates a config that has already
    # been through `_resolve_paths`.
    data_raw: str | Path
    data_interim: str | Path
    transcripts: str | Path
    translations: str | Path
    data_processed: str | Path
    data_external: str | Path
    logs: str | Path


class SoundCloudSection(_Section):
    profile_url: str
    broadcast_slug: str
    audio_codec: str
    rate_limit_seconds: float


class AsrSection(_Section):
    wav2vec2_model: str
    whisper_model_size: str
    somali_whisper_model: str
    elevenlabs_model: str
    gemini_model: str
    batch_size: int = Field(gt=0)
    use_fp16: bool
    chunk_length_s: int = Field(gt=0)
    chunk_overlap_ratio: float = Field(ge=0, lt=1)
    target_sample_rate: int = Field(gt=0)
    batch_size_days: int = Field(gt=0)


class OutputsSection(_Section):
    transcriptions_csv: str
    transcription_log: str


class TranslationSection(_Section):
    models: dict[str, str]
    src_lang: str
    tgt_lang: str
    text_column: str
    max_tokens: int = Field(gt=0)
    overlap_sentences: int = Field(ge=0)


class TopicsSection(_Section):
    spacy_model: str
    prob_threshold: float = Field(ge=0, le=1)
    bertopic: dict[str, object]
    theme_map: dict[int, str]
    theme_keywords: dict[str, list[str]]

    @model_validator(mode="after")
    def _themes_have_keywords(self):
        missing = set(self.theme_map.values()) - set(self.theme_keywords)
        if missing:
            raise ValueError(
                f"topics.theme_keywords is missing {sorted(missing)}. Without them the "
                "theme-drift guard silently skips those themes."
            )
        return self


class GeoSection(_Section):
    area_col: str
    fuzzy_score_cutoff: int = Field(ge=0, le=100)
    phase_colors: dict[int, str]


class RetrySection(_Section):
    count: int = Field(gt=0)
    delay_seconds: float = Field(ge=0)


class FeedbackSection(_Section):
    impact_signals: dict[str, list[str]]
    negators: list[str]
    negation_window: int = Field(ge=0)
    thresholds: dict[str, int]
    phase_effects: dict[str, int]

    @model_validator(mode="after")
    def _thresholds_match_effects(self):
        if set(self.thresholds) != set(self.phase_effects):
            raise ValueError(
                "feedback.thresholds and feedback.phase_effects must cover the same "
                f"signals; got {sorted(self.thresholds)} vs {sorted(self.phase_effects)}."
            )
        expected = {f"{signal}s" for signal in self.impact_signals}
        if set(self.thresholds) != expected:
            raise ValueError(
                "feedback.thresholds keys must be the plural of every impact signal; "
                f"expected {sorted(expected)}, got {sorted(self.thresholds)}."
            )
        return self


class Config(_Section):
    project: ProjectSection
    paths: PathsSection
    soundcloud: SoundCloudSection
    asr: AsrSection
    outputs: OutputsSection
    translation: TranslationSection
    topics: TopicsSection
    geo: GeoSection
    retry: RetrySection
    feedback: FeedbackSection


def validate(config: dict) -> None:
    """Raise :class:`pydantic.ValidationError` if *config* does not match the schema."""
    Config.model_validate(config)

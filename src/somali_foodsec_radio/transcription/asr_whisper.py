"""OpenAI Whisper and HuggingFace Somali-Whisper transcription engines.

Slimmed from notebooks/02 (the ASR model-comparison notebook): each engine loads its
model once and transcribes audio files. The original notebook's project-root walking
and self-installing ``pip`` calls are dropped — dependencies are declared in the
``[asr]`` extra instead.

Imports ``torch`` at load time; import this module by its full path only when
transcribing.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..config import get_setting
from ..logging_utils import get_logger

logger = get_logger(__name__)

WHISPER_MODEL_SIZES = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
]
# `somali` is overridable via `asr.somali_whisper_model`; the others are fixed
# upstream ids used for comparison runs.
SOMALI_WHISPER_MODELS = {
    "somali": "steja/whisper-small-somali",
    "multilingual": "openai/whisper-small",
    "large": "openai/whisper-large-v3",
}


def _select_device(use_gpu: bool) -> str:
    """Return ``"cuda"`` if a GPU is requested and available, else ``"cpu"``."""
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class WhisperEngine:
    """OpenAI Whisper transcription engine.

    The model is loaded lazily on the first :meth:`transcribe` call.
    """

    def __init__(self, model_size: str | None = None, use_gpu: bool = True):
        model_size = model_size or get_setting("asr.whisper_model_size", "small")
        if model_size not in WHISPER_MODEL_SIZES:
            logger.warning(
                "Unknown Whisper model size '%s'; using 'small'.", model_size
            )
            model_size = "small"
        self.model_size = model_size
        self.device = _select_device(use_gpu)
        self._model = None

    def _load(self):
        if self._model is None:
            import whisper

            logger.info(
                "Loading OpenAI Whisper '%s' on %s", self.model_size, self.device
            )
            self._model = whisper.load_model(self.model_size, device=self.device)
        return self._model

    def transcribe(self, audio_path: str | Path, language: str = "so") -> str:
        """Transcribe an audio file and return the transcript text."""
        model = self._load()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        result = model.transcribe(
            str(audio_path),
            language=language,
            temperature=0.2,
            word_timestamps=True,
            verbose=False,
            fp16=(self.device == "cuda"),
        )
        return result["text"].strip()


class SomaliWhisperEngine:
    """HuggingFace Somali-specialised Whisper engine (``steja/whisper-small-somali``).

    The pipeline is loaded lazily on the first :meth:`transcribe` call.
    """

    def __init__(self, model_type: str = "somali", use_gpu: bool = True):
        if model_type not in SOMALI_WHISPER_MODELS:
            logger.warning("Unknown model type '%s'; using 'somali'.", model_type)
            model_type = "somali"
        self.model_type = model_type
        self.model_id = (
            get_setting("asr.somali_whisper_model", SOMALI_WHISPER_MODELS["somali"])
            if model_type == "somali"
            else SOMALI_WHISPER_MODELS[model_type]
        )
        self.device = _select_device(use_gpu)
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                pipeline,
            )

            logger.info("Loading HF Whisper '%s' on %s", self.model_id, self.device)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id)
            processor = AutoProcessor.from_pretrained(self.model_id)
            model.to(self.device)
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=16 if self.device == "cuda" else 8,
                device=self.device,
            )
        return self._pipeline

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe an audio file and return the transcript text."""
        transcriber = self._load()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        result = transcriber(str(audio_path))
        text = (
            result["text"]
            if isinstance(result, dict) and "text" in result
            else str(result)
        )
        return text.strip()

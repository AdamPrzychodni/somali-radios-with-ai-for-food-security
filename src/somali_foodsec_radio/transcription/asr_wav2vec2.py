"""GPU-optimised Somali speech-to-text using a Wav2Vec2 CTC model.

This module imports ``torch``/``transformers``/``librosa`` at module load time, so it
is only imported lazily by the code that actually transcribes audio (it requires the
``[asr]`` optional dependencies).
"""

from __future__ import annotations

import io
import re

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class SomaliASREngine:
    """GPU-optimised transcription engine with batched inference.

    Uses L4 Tensor Cores (FP16) for a 3-5x speed-up over CPU.
    """

    def __init__(
        self,
        model_name: str = "Mustafaa4a/ASR-Somali",
        batch_size: int = 8,
        use_fp16: bool = True,
        verbose: bool = False,
    ):
        """Initialise the GPU-optimised ASR model.

        Args:
            model_name: HuggingFace model identifier.
            batch_size: Number of audio chunks to process simultaneously.
            use_fp16: Enable mixed precision for faster inference on L4 GPUs.
            verbose: Enable detailed logging.
        """
        self.processor: Wav2Vec2Processor | None = None
        self.model: Wav2Vec2ForCTC | None = None
        self.device: str = "cpu"
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.verbose = verbose
        self._setup_model(model_name)

    def _setup_model(self, model_name: str) -> None:
        """Load the model with GPU optimisations.

        Raises:
            RuntimeError: If the model fails to load.
        """
        if self.verbose:
            print(f"Loading ASR model: {model_name}")

        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            if self.device == "cuda":
                self.model.to(self.device)

                # Enable mixed precision on L4 (Tensor Cores).
                if self.use_fp16:
                    self.model = self.model.half()
                    if self.verbose:
                        print("✓ FP16 mixed precision enabled")

                self.model.eval()
                torch.backends.cudnn.benchmark = True

                if self.verbose:
                    gpu_name = torch.cuda.get_device_name(0)
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"✓ Model loaded on GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
                    print(f"✓ Batch size: {self.batch_size}")
            else:
                self.model.to(self.device)
                if self.verbose:
                    print("⚠️  Running on CPU (slower)")
        except Exception as exc:
            raise RuntimeError(f"Failed to load ASR model: {exc}") from exc

    def transcribe_from_memory_batched(self, audio_data: bytes) -> str:
        """GPU-accelerated batched transcription of raw audio bytes.

        Args:
            audio_data: Raw audio bytes.

        Returns:
            Transcribed text.

        Raises:
            ValueError: If transcription fails.
            RuntimeError: If the model is not initialised.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("ASR model is not initialized.")

        target_sr = 16000

        try:
            audio, _ = librosa.load(io.BytesIO(audio_data), sr=target_sr)

            # Larger 60s chunks for GPU batch processing on L4.
            chunk_length_s = 60
            chunk_length = chunk_length_s * target_sr
            overlap = int(chunk_length * 0.1)  # 10% overlap for accuracy

            chunks = []
            for i in range(0, len(audio), chunk_length - overlap):
                chunk = audio[i : i + chunk_length]
                if len(chunk) > target_sr:  # skip chunks shorter than 1 second
                    chunks.append(chunk)

            if not chunks:
                return ""

            transcriptions = []
            for batch_start in range(0, len(chunks), self.batch_size):
                batch_chunks = chunks[batch_start : batch_start + self.batch_size]

                inputs = self.processor(
                    batch_chunks,
                    sampling_rate=target_sr,
                    return_tensors="pt",
                    padding=True,
                )
                input_values = inputs.input_values.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)

                with torch.no_grad():
                    if self.use_fp16 and self.device == "cuda":
                        input_values = input_values.half()
                    logits = self.model(
                        input_values, attention_mask=attention_mask
                    ).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                transcriptions.extend(self.processor.batch_decode(predicted_ids))

            full_text = " ".join(transcriptions)
            return re.sub(r"\s+", " ", full_text).strip()
        except Exception as exc:
            raise ValueError(f"Transcription failed: {exc}") from exc

    def get_vram_usage(self) -> dict[str, float]:
        """Return current GPU memory usage in GB (zeros on CPU)."""
        if self.device == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            }
        return {"allocated_gb": 0, "reserved_gb": 0}

"""HuggingFace translation models (NLLB and MADLAD) for Somali -> English.

This module imports ``torch``/``transformers`` at load time, so import it by its full
path only when translating (it requires the ``[analysis]`` optional dependencies).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from .chunking import create_semantic_chunks, deduplicate_overlap

# Map NLLB language codes to MADLAD-style <2xx> prefixes.
NLLB_TO_MADLAD_CODE: dict[str, str] = {
    "eng_Latn": "<2en>",
    "som_Latn": "<2so>",
}


def load_model(model_name: str, model_type: str) -> tuple[Any, Any, Any]:
    """Load a translation model and tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        model_type: ``'nllb'`` or ``'madlad'``.

    Returns:
        ``(tokenizer, model, device)``.
    """
    print(f"Loading {model_name} (type: {model_type})...")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if model_type == "nllb":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(
            device_str
        )
        device = device_str
        print(f"Model loaded on device: {device}")

    elif model_type == "madlad":
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, device_map="auto", torch_dtype=dtype
        )
        device = model.device
        print(f"Model loaded with device_map='auto'. Main device: {device}")

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'nllb' or 'madlad'.")

    model.eval()
    return tokenizer, model, device


def translate_text_chunked(
    text: str,
    tokenizer: Any,
    model: Any,
    device: Any,
    model_type: str,
    src_lang: str,
    tgt_lang: str,
    max_tokens: int = 450,
    overlap_sentences: int = 2,
) -> str:
    """Translate long *text* by splitting it into token-bounded chunks."""
    if not text or not isinstance(text, str):
        print("  -> Warning: Empty or invalid text provided. Returning empty string.")
        return ""

    chunks = create_semantic_chunks(
        text=text,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap_sentences=overlap_sentences,
    )
    print(f"  -> Split into {len(chunks)} chunks for translation")

    translated_chunks: list[str] = []

    if model_type == "nllb":
        tokenizer.src_lang = src_lang
        tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        for idx, (chunk_text, _, _) in enumerate(chunks):
            if not chunk_text.strip():
                print(f"  -> Skipping empty chunk {idx + 1}/{len(chunks)}")
                continue

            inputs = tokenizer(
                chunk_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(device)

            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=512,
                    num_beams=5,
                    length_penalty=1.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

            translation = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]
            translated_chunks.append(translation)
            print(
                f"  -> Translated chunk {idx + 1}/{len(chunks)} "
                f"({len(translation)} chars)"
            )

    elif model_type == "madlad":
        try:
            target_prefix = NLLB_TO_MADLAD_CODE[tgt_lang]
        except KeyError:
            print(f"Error: No MADLAD code found for '{tgt_lang}'.")
            return "[Translation Error: Unknown target language]"

        for idx, (chunk_text, _, _) in enumerate(chunks):
            if not chunk_text.strip():
                print(f"  -> Skipping empty chunk {idx + 1}/{len(chunks)}")
                continue

            prefixed_text = f"{target_prefix} {chunk_text}"
            inputs = tokenizer(
                prefixed_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(model.device)

            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs, max_length=512, num_beams=5, early_stopping=True
                )

            translation = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]
            translated_chunks.append(translation)
            print(
                f"  -> Translated chunk {idx + 1}/{len(chunks)} "
                f"({len(translation)} chars)"
            )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return deduplicate_overlap(translated_chunks)


def batch_translate_dataframe(
    df: pd.DataFrame,
    text_column: str,
    tokenizer: Any,
    model: Any,
    device: Any,
    model_type: str,
    src_lang: str,
    tgt_lang: str,
    new_column_name: str = "transcript_text_english",
    use_chunking: bool = True,
) -> pd.DataFrame:
    """Translate *text_column* of *df*, writing results into *new_column_name*."""
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in DataFrame")

    df_translated = df.copy()
    translations: list[Any] = []

    print(f"Translating {len(df_translated)} rows using {model_type}...")

    target_prefix = ""
    tgt_lang_id = None

    if model_type == "nllb":
        tokenizer.src_lang = src_lang
        tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    elif model_type == "madlad":
        try:
            target_prefix = NLLB_TO_MADLAD_CODE[tgt_lang]
        except KeyError as exc:
            raise ValueError(f"MADLAD code for '{tgt_lang}' not defined.") from exc

    for idx, row in df_translated.iterrows():
        text = row[text_column]
        try:
            positional_idx = df_translated.index.get_loc(idx) + 1
            print(f"\nRow {positional_idx}/{len(df_translated)} (Index: {idx})")

            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                print("  -> Skipping row: Text is missing or empty.")
                translations.append(None)
                continue

            if use_chunking:
                translation = translate_text_chunked(
                    text=text,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    model_type=model_type,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                )
            else:
                with torch.no_grad():
                    if model_type == "nllb":
                        inputs = tokenizer(
                            text, return_tensors="pt", max_length=512, truncation=True
                        ).to(device)
                        translated_tokens = model.generate(
                            **inputs,
                            forced_bos_token_id=tgt_lang_id,
                            max_length=512,
                        )
                    elif model_type == "madlad":
                        prefixed_text = f"{target_prefix} {text}"
                        inputs = tokenizer(
                            prefixed_text,
                            return_tensors="pt",
                            max_length=512,
                            truncation=True,
                        ).to(model.device)
                        translated_tokens = model.generate(**inputs, max_length=512)
                    translation = tokenizer.batch_decode(
                        translated_tokens, skip_special_tokens=True
                    )[0]

            translations.append(translation)
            print(f"✓ Completed row {positional_idx}")
        except Exception as exc:  # noqa: BLE001 - record failure, continue with the rest
            print(f"✗ Error translating row {idx}: {exc}")
            translations.append(None)

    df_translated[new_column_name] = translations
    return df_translated

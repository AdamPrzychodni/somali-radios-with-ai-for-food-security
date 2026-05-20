"""End-to-end translation pipeline: load data, load model, translate, free VRAM.

Imports ``torch`` at load time — import this module by its full path only when
translating (it requires the ``[analysis]`` optional dependencies).
"""

from __future__ import annotations

import gc
from typing import Optional, Tuple

import pandas as pd
import torch

from .chunking import create_semantic_chunks
from .translate_hf import batch_translate_dataframe, load_model


def load_transcription_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load transcription data from a CSV file into a DataFrame (``None`` if missing)."""
    try:
        df = pd.read_csv(file_path)
        print("File loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at the path: {file_path}")
        return None


def run_translation_pipeline(
    df: pd.DataFrame,
    model_name: str,
    model_type: str,
    src_lang: str = "som_Latn",
    tgt_lang: str = "eng_Latn",
    num_rows_to_translate: Optional[int] = None,
    text_column: str = "transcript_text",
) -> Optional[Tuple[pd.DataFrame, str]]:
    """Load a model, translate *text_column*, free VRAM, and return the result.

    Returns ``(translated_dataframe, new_column_name)`` on success, ``None`` on failure.
    """
    print("\n" + "#" * 80)
    print(f"STARTING PIPELINE FOR: {model_name} (Type: {model_type})")
    print("#" * 80)

    tokenizer, model, device = None, None, None
    test_df_translated = None
    new_col_name = f"translated_{model_type}_{model_name.split('/')[-1]}"

    try:
        # 1. Load model.
        tokenizer, model, device = load_model(model_name, model_type)

        # 2. Analyse chunking on the first row.
        print("\n" + "=" * 80)
        print(f"CHUNKING ANALYSIS ({model_name})")
        print("=" * 80)
        sample_text = df.iloc[0][text_column]
        if pd.isna(sample_text):
            print("Sample text in row 0 is NA. Skipping chunk analysis.")
        else:
            print(
                f"Sample text: {len(str(sample_text))} chars, "
                f"{len(str(sample_text).split())} words\n"
            )
            chunks = create_semantic_chunks(
                sample_text, tokenizer, max_tokens=450, overlap_sentences=2
            )
            if chunks:
                print(f"\nFinal chunk count: {len(chunks)}")
                avg_chunk_chars = sum(len(c[0]) for c in chunks) / len(chunks)
                print(f"Average chunk size: {avg_chunk_chars:.0f} chars")
            else:
                print("No chunks were created for the sample text.")
        print("=" * 80)

        # 3. Select rows for translation.
        if num_rows_to_translate is None:
            print(f"\nPreparing to translate all {len(df)} rows...")
            df_to_translate = df.copy()
        else:
            print(f"\nPreparing to translate first {num_rows_to_translate} rows...")
            df_to_translate = df.head(num_rows_to_translate).copy()

        # 4. Run batch translation.
        test_df_translated = batch_translate_dataframe(
            df=df_to_translate,
            text_column=text_column,
            tokenizer=tokenizer,
            model=model,
            device=device,
            model_type=model_type,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            new_column_name=new_col_name,
            use_chunking=True,
        )

        # 5. Display results.
        print("\n" + "=" * 80)
        print(f"TRANSLATION RESULTS ({model_name})")
        print("=" * 80)
        if not test_df_translated.empty:
            for idx, row in test_df_translated.iterrows():
                original = (
                    str(row[text_column]) if pd.notna(row[text_column]) else ""
                )
                translated = (
                    str(row[new_col_name]) if pd.notna(row[new_col_name]) else ""
                )
                print(f"\n--- Row {idx} ---")
                print(f"Original: {len(original)} chars")
                print(f"Translated: {len(translated)} chars")
                print(f"\nFirst 500 chars (English):\n{translated[:500]}...")
            print("=" * 80)

        # 6. Return the result on success.
        return test_df_translated, new_col_name

    except Exception as exc:  # noqa: BLE001 - pipeline-level failure handler
        print(f"ERROR: Pipeline failed for {model_name}.")
        print(f"Details: {exc}")
        return None

    finally:
        # 7. Free VRAM.
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\nCleaned up model {model_name} from memory (VRAM cleared).")
        print("#" * 80 + "\n")

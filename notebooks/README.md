# Notebooks

These notebooks are thin **drivers**: the reusable logic lives in the
`somali_foodsec_radio` package, and each notebook just imports it and runs a
pipeline stage. Install the package first, from the repo root:

```bash
pip install -e '.[all]'
```

Model ids, paths and thresholds come from `config/config.yaml`; API keys come
from a `.env` file (copy `.env.example`).

## Run order

| #  | Notebook                        | Stage                                                        |
|----|---------------------------------|--------------------------------------------------------------|
| 01 | `01_data_collection`            | Download Radio Ergo broadcasts from SoundCloud → `data/raw/` |
| 02 | `02_asr_model_comparison`       | Benchmark speech-to-text models (experimental)               |
| 03 | `03_transcription`              | Transcribe audio with ElevenLabs Scribe / Gemini             |
| 04 | `04_translation`                | Translate transcripts Somali → English (NLLB / MADLAD)       |
| 05 | `05_eda_ipc`                    | Explore IPC food-security data, plot phase maps              |
| 06 | `06_topic_modeling`             | Extract themes, pair them with IPC geographies               |
| 07 | `07_ipc_update_from_feedback`   | Adjust IPC phases from Radio Ergo caller-feedback PDFs        |

The pipeline has two streams that both feed the IPC update:

```
01 → 03 → 04 → 06 ┐
                  ├→ IPC food-security phases
07 (feedback PDFs)┘
05 explores the IPC baseline data that 06 and 07 build on.
```

## Notes

- **Notebook 02** is an experimental model-comparison notebook. It keeps its own
  self-contained transcription code rather than importing the package — it is a
  benchmark record, not part of the production pipeline. The production engines
  live in `somali_foodsec_radio.transcription`.
- Cell outputs are not committed; re-run a notebook to regenerate them.

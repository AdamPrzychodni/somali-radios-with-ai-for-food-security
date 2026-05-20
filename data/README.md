# Data

The pipeline reads and writes here. **Contents are gitignored** — only this README and
the `.gitkeep` placeholders are tracked, so the folder layout survives a fresh clone.

| Folder                   | Holds                                                              |
|--------------------------|--------------------------------------------------------------------|
| `raw/`                   | Downloaded broadcast audio (`.mp3`) from SoundCloud.               |
| `interim/transcripts/`   | Somali transcripts + `transcriptions_database.csv`.                |
| `interim/translations/`  | English translations of the transcripts.                          |
| `processed/`             | Analysis outputs: theme-location pairs, weekly impact, updated IPC.|
| `external/`              | Third-party inputs: IPC GeoJSON files, Radio Ergo feedback PDFs.   |

Stages map to the pipeline: `raw` → transcription → `interim/transcripts` → translation →
`interim/translations` → topic modelling / feedback analysis → `processed`.

Paths are configurable in `config/config.yaml` (the `paths:` section).

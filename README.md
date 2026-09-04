# 📻 Leveraging Local Radio for Real-Time Food Security Insights 🇸🇴

[![CI](https://github.com/AdamPrzychodni/somali-radios-with-ai-for-food-security/actions/workflows/ci.yml/badge.svg)](https://github.com/AdamPrzychodni/somali-radios-with-ai-for-food-security/actions/workflows/ci.yml)

An AI pipeline, in collaboration with **Zero Hunger Lab**, that turns Somali radio
broadcasts (Radio Ergo) into food-security insights. Broadcasts are downloaded,
transcribed, translated, mined for themes, and combined with caller-feedback reports
to adjust **IPC** (Integrated Phase Classification) food-security phases.

---

## Pipeline

```
01 collect ─→ 03 transcribe ─→ 04 translate ─→ 06 topic-model ─┐
                                                               ├─→ IPC phase update
       Radio Ergo feedback PDFs ─→ 07 impact signals ──────────┘
       05 explores the IPC baseline data the analysis builds on.
```

| Stage         | What it does                                              | Models / tools                          |
|---------------|-----------------------------------------------------------|------------------------------------------|
| Collection    | Download broadcasts from SoundCloud                       | `yt-dlp`                                 |
| Transcription | Speech-to-text (Somali)                                   | Wav2Vec2 `Mustafaa4a/ASR-Somali`, Whisper, ElevenLabs Scribe, Gemini |
| Translation   | Somali → English                                          | NLLB-200, MADLAD-400, Gemini             |
| Topic model   | Themes + locations from transcripts                       | BERTopic, spaCy                          |
| Feedback      | Parse caller-feedback PDFs → impact signals → IPC phases  | `pdfplumber`, `rapidfuzz`, GeoPandas     |

---

## Project structure

```
somali-radios-with-ai-for-food-security/
├── config/
│   └── config.yaml              # paths, model ids, thresholds (no secrets)
├── data/                        # raw / interim / processed / external (gitignored)
├── notebooks/                   # 7 pipeline notebooks — see notebooks/README.md
├── src/somali_foodsec_radio/    # the package
│   ├── collection/              # SoundCloud download
│   ├── transcription/           # speech-to-text engines + batch runner
│   ├── translation/             # Somali → English
│   ├── topics/                  # BERTopic theme extraction
│   ├── geo/                     # IPC geometries + location matching
│   ├── feedback/                # caller-feedback → IPC phase updates
│   ├── config.py / paths.py     # config loading, path resolution
│   └── cli.py                   # the `radio-collect` console script
├── docs/plan.md                 # roadmap, ASR research, open questions
├── tests/                       # pytest suite (pure-logic functions)
├── pyproject.toml               # package metadata + dependencies + ruff config
├── uv.lock                      # pinned, reproducible dependency set
└── .env.example                 # template for API keys
```

---

## Installation

Requires **Python 3.11+** and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Clone
git clone https://github.com/AdamPrzychodni/somali-radios-with-ai-for-food-security.git
cd somali-radios-with-ai-for-food-security

# 2. Install. uv creates .venv and installs the locked versions from uv.lock.
uv sync                       # core + dev tooling; enough to run the test suite
uv sync --all-extras          # everything, including torch and geopandas
#   or only what you need:
#   uv sync --extra asr       # audio download + speech-to-text
#   uv sync --extra analysis  # translation, topics, geo, PDF feedback
#   uv sync --extra apis      # Gemini / ElevenLabs API clients

# 3. spaCy model (needed for topic modelling)
uv run python -m spacy download en_core_web_sm

# 4. ffmpeg (needed for audio processing)
#   macOS:  brew install ffmpeg
#   Ubuntu: sudo apt-get install ffmpeg

# 5. API keys (only for the Gemini / ElevenLabs engines)
cp .env.example .env               # then fill in your keys

# 6. Git hooks (lint, format, strip notebook outputs)
uv run pre-commit install
```

`uv.lock` pins every transitive dependency, so a rerun months from now
reproduces today's numbers rather than whatever `transformers` ships next.

---

## Usage

### Command line

Download and transcribe a date range of broadcasts with the GPU-optimised
Wav2Vec2 engine:

```bash
uv run radio-collect --start 2022-01-01 --end 2022-03-31 \
    --output data/interim/transcripts --verbose
```

`uv run radio-collect --help` lists all options.

### Notebooks

The seven notebooks drive the pipeline stage by stage. Start Jupyter and follow
the run order in [`notebooks/README.md`](notebooks/README.md):

```bash
uv run jupyter notebook
```

Each notebook is a thin driver — the real logic lives in the
`somali_foodsec_radio` package, so notebooks stay short and the same code is
reused, tested, and importable.

### As a library

```python
from somali_foodsec_radio.collection import download_radio_ergo_by_date
from somali_foodsec_radio.translation.pipeline import run_translation_pipeline
from somali_foodsec_radio.feedback import create_impact_signals
```

---

## Configuration

- **`config/config.yaml`** — paths, model ids, chunking params, thresholds, theme
  maps, keyword lists. Loaded and **schema-validated** by
  `somali_foodsec_radio.config`, so a typo fails at load instead of twenty minutes
  into a GPU run. Every key is read by the code; unknown keys are rejected. Copy
  `config/config.local.yaml.example` to `config.local.yaml` (gitignored) to
  override any setting locally.
- **`.env`** — API keys (`GEMINI_API_KEY`, `ELEVENLABS_API_KEY`). Never committed.

Outputs are stamped with `model_id`, `config_hash`, `package_version` and a
timestamp, plus a `.run.json` sidecar — so any result can be traced back to the
run and configuration that produced it.

---

## Testing and linting

```bash
uv run pytest          # 114 tests, ~1 s
uv run ruff check .    # lint
uv run ruff format .   # format
```

The suite covers the pure-logic functions (URL parsing, text chunking, location
matching, negation-aware signal detection, IPC phase math, theme-drift detection,
output provenance) — no GPU, network or API keys required. CI runs all three on
every push.

---

## Data privacy & ethics

- All processing respects copyright and fair-use guidelines.
- No personal data is collected from radio broadcasts.
- API usage follows the respective providers' terms of service.
- Research is conducted under academic ethics protocols.

---

## Where this is going

[`docs/plan.md`](docs/plan.md) is the single planning document: current state,
the measurement layer each stage still needs, the transcription model gate
(including the [PazaBench](https://huggingface.co/spaces/microsoft/paza-bench)
Somali leaderboard), and the open questions.

The short version: the code is good, the results are unproven. No stage has a
number attached to it yet, and the headline claim — that caller feedback improves
IPC phase estimates — has never been checked against "leave the phase unchanged".

---

## Contributing

This is an academic research project.

- **Contact**: Adam Przychodni
- **Institution**: Zero Hunger Lab collaboration
- **Purpose**: Food-security research and humanitarian applications

---

## License

Released under the [MIT License](LICENSE). Please respect the copyright of the
original radio content and the terms of service of any APIs you use.

# 📻 Leveraging Local Radio for Real-Time Food Security Insights 🇸🇴

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
├── tests/                       # pytest suite (pure-logic functions)
├── pyproject.toml               # package metadata + dependencies
└── .env.example                 # template for API keys
```

---

## Installation

Requires **Python 3.10+**.

```bash
# 1. Clone
git clone https://github.com/AdamPrzychodni/somali-radios-with-ai-for-food-security.git
cd somali-radios-with-ai-for-food-security

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install the package (editable)
pip install -e '.[all]'            # everything
#   or install only what you need:
#   pip install -e '.[asr]'        # audio download + speech-to-text
#   pip install -e '.[analysis]'   # translation, topics, geo, PDF feedback
#   pip install -e '.[apis]'       # Gemini / ElevenLabs API clients

# 4. spaCy model (needed for topic modelling)
python -m spacy download en_core_web_sm

# 5. ffmpeg (needed for audio processing)
#   macOS:  brew install ffmpeg
#   Ubuntu: sudo apt-get install ffmpeg

# 6. API keys (only for the Gemini / ElevenLabs engines)
cp .env.example .env               # then fill in your keys
```

---

## Usage

### Command line

Download and transcribe a date range of broadcasts with the GPU-optimised
Wav2Vec2 engine:

```bash
radio-collect --start 2022-01-01 --end 2022-03-31 \
    --output data/interim/transcripts --verbose
```

`radio-collect --help` lists all options.

### Notebooks

The seven notebooks drive the pipeline stage by stage. Start Jupyter and follow
the run order in [`notebooks/README.md`](notebooks/README.md):

```bash
jupyter notebook
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
  maps. Loaded by `somali_foodsec_radio.config`. Copy
  `config/config.local.yaml.example` to `config.local.yaml` (gitignored) to
  override any setting locally.
- **`.env`** — API keys (`GEMINI_API_KEY`, `ELEVENLABS_API_KEY`). Never committed.

---

## Testing

```bash
pytest
```

The suite covers the pure-logic functions (URL parsing, text chunking, location
matching, impact-signal detection, IPC phase math) — no GPU, network or API keys
required.

---

## Data privacy & ethics

- All processing respects copyright and fair-use guidelines.
- No personal data is collected from radio broadcasts.
- API usage follows the respective providers' terms of service.
- Research is conducted under academic ethics protocols.

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

"""somali_foodsec_radio — turn Somali radio broadcasts into food-security insights.

The package is organised by pipeline stage:

* :mod:`somali_foodsec_radio.collection`    — download broadcasts from SoundCloud
* :mod:`somali_foodsec_radio.transcription` — speech-to-text engines (Wav2Vec2, Whisper,
  ElevenLabs Scribe, Gemini)
* :mod:`somali_foodsec_radio.translation`   — Somali -> English translation
* :mod:`somali_foodsec_radio.topics`        — topic modelling and theme extraction
* :mod:`somali_foodsec_radio.geo`           — IPC geometries and location matching
* :mod:`somali_foodsec_radio.feedback`      — Radio Ergo caller-feedback -> IPC updates

Importing this package is cheap: heavy ML dependencies (torch, transformers, bertopic,
spacy, geopandas, ...) are imported lazily inside the functions that need them.
"""

__version__ = "0.1.0"

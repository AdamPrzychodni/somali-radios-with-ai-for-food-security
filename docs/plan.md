# Project Plan — Somali Radio → Food Security

> The single planning document for this repository. Supersedes the earlier
> `plans/improvement-plan.md`, `plans/asr-improvement-roadmap.md` and
> `plans/asr-research-brief.md`, which are merged here.
>
> Last updated **2026-09-04**.

## Contents

1. [The one-sentence version](#1-the-one-sentence-version)
2. [Where the project stands](#2-where-the-project-stands)
3. [Phase 0 — stop being silently wrong](#3-phase-0--stop-being-silently-wrong)
4. [Phase 1 — does the approach work at all?](#4-phase-1--does-the-approach-work-at-all)
5. [Phase 2 — the transcription gate](#5-phase-2--the-transcription-gate)
6. [Sequencing](#6-sequencing)
7. [Open questions](#7-open-questions)
8. [Appendix A — ASR evidence base](#appendix-a--asr-evidence-base)

---

## 1. The one-sentence version

**The code is now good; the results are unproven.** The 2026-05 refactor turned a notebook pile
into a tested package — but no stage of the pipeline has a number attached to it, and the last
stage, the one that carries the project's actual claim, rests on hand-picked keywords and
magic-number thresholds that have never been checked against a real IPC outcome.

Everything below follows from that, plus a handful of bugs that made results silently wrong.

---

## 2. Where the project stands

| | |
|---|---|
| **Code quality** | Good. `src/somali_foodsec_radio/` is a clean package, one subpackage per pipeline stage, ~3.2k lines. |
| **Tests** | Pure logic only — chunking, URL parsing, phase math, signal detection. Fast (~1.3 s). |
| **Measurement** | **None, at any stage.** No gold set, no metric, no baseline, anywhere in the repo. |
| **Data** | **`data/` is empty.** The 2020–2025 archive, the IPC GeoJSON and the feedback PDFs all need re-collecting. |
| **Reproducibility** | Addressed in Phase 0: seeds, a lockfile, CI and output provenance did not exist before. |

The pipeline is a chain: **collect → transcribe → translate → topic-model → adjust IPC phase**,
plus a parallel caller-feedback-PDF stream that feeds the same IPC update.

---

## 3. Phase 0 — stop being silently wrong

*This is the pass executed on 2026-09-04. Recorded here so the reasoning survives.*

### 3.1 Bugs that made results wrong

**"Rainfall positive" fired on drought reports.** `feedback/signals.py` matched `\brain\b` with no
negation handling, so *"no rain"*, *"the rains failed"* and *"still waiting for rain"* all set
`rainfall_positive = 1` — worth **−1 IPC phase**. The pipeline reported a food-security
*improvement* on a drought report. Same class of error: `death` → `livestock_disease` matched
human-casualty reports, and `dry` → `drought_warning` matched "dry season".

**Topic ids were unstable, so theme labels drifted.** `config.yaml` maps
`theme_map: {0: Rainfall, 1: Crop Failure, …}` and themes were resolved *by position in the
probability vector*. BERTopic assigns ids by cluster size, and no random seed existed anywhere in
the repo — so a rerun reshuffled the ids and "Rainfall" quietly became a different topic. Nothing
detected it.

**Half of `config.yaml` was dead — including every scientific parameter.** The
`feedback.impact_signals` / `thresholds` / `phase_effects` values were duplicated as hardcoded
module constants, and the notebooks called the functions without passing config. Editing
`config.yaml` changed nothing: someone tuning thresholds would get identical output and conclude
the parameters did not matter. The same applied to `retry.*`, `asr.chunk_length_s` and several
`soundcloud.*` / `geo.*` keys.

**The weekly IPC loop never accumulated.** The original baseline `geo_df` went into every
iteration, so each week was scored against the same static starting point. This is defensible as
a *per-week deviation from baseline* and is now documented as such — but it was undocumented and
the notebook's framing implied a trajectory.

**Smaller ones.** `adjust_ipc_phases` subtracted raw event *counts* from a *phase* (a 12-event
week saturated at phase 5) — units did not match, and it was superseded by the threshold variant.
`assign_geography` returned one area per broadcast, so a bulletin covering three regions was
attributed to one, and it used `"Unknown"` where its sibling used `None`. `plot_time_series` put
matplotlib inside a domain-logic module.

### 3.2 Engineering practice

- **uv + a committed `uv.lock`.** `transformers` / `bertopic` / `torch` move fast enough to change
  results; without a lockfile a rerun in six months reproduces nothing.
- **ruff** for lint and format, **pre-commit** to run it, **`nbstripout`** so notebook outputs stop
  being committed.
- **GitHub Actions CI** — ruff + pytest on push and PR.
- **A seed** (`project.seed`) threaded into UMAP/BERTopic and any sampling.
- **Run provenance** — every output CSV is stamped with the model id, a config hash, the package
  version and a timestamp, with a sidecar `run.json`. Without it no eval set can be traced back to
  the run that produced it.
- **Config schema validation** — a malformed `config.yaml` now fails at load with a clear message
  instead of a `KeyError` twenty minutes into a GPU run.

### 3.3 Repo hygiene

`origin/main` was 9 commits behind the refactor, so anyone cloning the repo got the old notebook
pile. Fast-forwarded and pushed; the two stale duplicate branches are gone. This document replaces
the untracked `plans/` directory.

---

## 4. Phase 1 — does the approach work at all?

**Run the cheap end-to-end check before improving any single ingredient.**

Better transcription only helps if the rest of the chain converts better transcripts into better
IPC calls. Nobody has checked that the chain converts anything at all. So:

> **Do the radio-adjusted IPC phases predict the next official IPC classification better than
> leaving the phase unchanged?**

- **Data**: historical IPC classifications plus the feedback PDFs. No new transcription needed —
  but both need re-collecting first (§2).
- **Method**: walk-forward by week; score with a proper scoring rule; compare against the
  unchanged-baseline permanent reference; report skill, calibration, and the decision loss at the
  threshold that matters (does the area cross into Phase 3+?).
- **Also cheap and worth doing here**: hand-label ~200 call remarks for the five signals and
  measure keyword precision/recall. That tells you whether the negation bug was a rounding error
  or the dominant error term.
- **Effort**: ~3–4 days. **Output: a yes/no on whether the approach has signal.**

"No change" is the permanent baseline. Anything the pipeline produces has to beat it.

**Why this ordering matters.** If yes, there is signal and the ASR work below becomes the obvious
way to amplify it — and it gains a *downstream* metric, so "is this ASR model better?" gets a
food-security answer, not just a CER answer. If no, better transcripts would not have rescued it,
and the effort goes to the adjustment rule instead of to a GPU.

### 4.1 The rest of the measurement layer

One gold set and one metric per stage, listed last-stage-first, which is the order to build them.

| Stage | Gold set | Metric |
|---|---|---|
| **IPC adjustment** | Historical IPC releases + feedback PDFs | Skill vs. no-change, calibration, decision loss at Phase 3+ |
| **Transcription** | ~30 clips × 30 s, stratified (see §5) | **CER** primary, WER secondary, keyterm F1, script fidelity |
| **Translation** | ~100 Somali→English sentence pairs from the *same* clips | chrF++ and COMET |
| **Topic model** | ~100 hand-labelled translated transcripts | Per-theme precision/recall — also catches theme drift on every rerun |

One transcription effort produces two eval sets (ASR and translation). Three translation models
are currently wired with no way to choose between them.

---

## 5. Phase 2 — the transcription gate

Transcription is the first AI stage and every downstream stage inherits its errors. Two facts
drive this section:

1. The model in production (`Mustafaa4a/ASR-Somali`) is **one of the weaker available options** —
   ~30.6 % WER on its *own* easy held-out set; stronger open models report roughly half the error
   on harder data.
2. **We cannot currently prove that, or anything else, about our transcripts.** There is no
   in-domain reference set and no error metric in the codebase.

### 5.1 Initiative A — evaluation harness · **P0, blocks everything**

Every other decision depends on being able to measure. Today's `02_asr_model_comparison.ipynb`
produces opinions (word-frequency and repetition heuristics), not numbers.

- An **in-domain gold reference set** — ~30 clips of ~30 s sampled across 2020–2025, stratified by
  year, segment type (studio anchor / field correspondent / phone caller / music intro) and topic.
  Hand-transcribed by a native Somali speaker, in verbatim *and* normalised versions. Stored under
  `data/external/` (gitignored) and documented so it is reproducible.
- A **metrics module** computing CER, WER, **keyterm F1** (against the gazetteer below) and a
  **script-fidelity check** flagging non-Latin output — Whisper's known Somali failure mode. Pure
  functions, so per repo convention they get their own module and unit tests; `jiwer` is the
  natural dependency.
- A **rewrite of `02_asr_model_comparison.ipynb`** into a real leaderboard.

*~2–3 days engineering + 2–3 days human transcription (or ~$200 contracted).*
**Output: a reproducible model leaderboard.**

### 5.2 Initiative B — benchmark stronger candidates · **P1**

**New since the May 2026 research: [PazaBench](https://huggingface.co/spaces/microsoft/paza-bench)**
(Microsoft Research, Feb 2026) — the first ASR leaderboard dedicated to low-resource languages:
39 African languages, 52 models, scored on CER / WER / RTFx, **Somali included**. Candidate
selection is no longer guesswork; read the Somali column, then confirm on our own audio.

| Candidate | Why |
|---|---|
| `microsoft/paza-whisper-large-v3-turbo` | Already Somali-fine-tuned; MSR benchmarks it against OmniASR-7B, MMS-1B and Whisper-v3-turbo. Probably the best ready-to-run open model. |
| `microsoft/paza-Phi-4-multimodal-instruct` | Same family, multimodal — new since the brief |
| `facebook/seamless-m4t-v2-large` | Strongest open zero-shot Somali (17.9 % CER FLEURS); **does Somali→English speech translation in one step**, so it could eventually collapse the transcribe and translate stages |
| `facebook/omniASR-LLM-7B` | 1,600+ languages, built for long audio |
| `facebook/mms-1b-all` | Lightweight, reliably Latin-script |
| ElevenLabs **Scribe v2** | The repo pins v1; v2 is the current batch model |
| Gemini (current release) | `config.yaml` pinned `gemini-2.0-flash`, two generations stale |
| Google Chirp 3 BatchRecognize | Managed fallback: zero-ops, but ~$1,300 for the full archive |

*~1 day per engine integration + GPU/API time; ~$50 of API spend for the managed options.*

### 5.3 Initiative C — VAD pre-processing · **P1**

Radio broadcasts carry music stings, silence and station IDs; stripping them typically removes
5–15 % of garbage and improves every engine. SeamlessM4T additionally *requires* ≤30 s segments.
A `silero-vad` or `pyannote` pre-pass replaces today's fixed 60 s chunking, with config keys for
enable/disable and thresholds. *~2–3 days.*

### 5.4 Initiative D — promote the winner · **P1, gated on B**

**Decision rule: CER ≤ 15 % and keyterm F1 ≥ 0.85 on the gold set → ship it; otherwise fine-tune.**
Update the `asr:` section of `config/config.yaml` and the streaming collector's default engine,
then re-run a sample to confirm. *~1–2 days.*

**Timing matters more than usual here.** The archive has to be re-collected from scratch, so the
engine should be chosen *before* the bulk run. Transcribing ~900 h twice is the expensive mistake
available to this project.

### 5.5 Initiative E — fine-tune on Radio Ergo audio · **P2, conditional**

Only if no off-the-shelf model clears the decision rule. Hand-transcribe 5–10 h of Radio Ergo
audio; LoRA fine-tune `whisper-large-v3-turbo` starting from the `paza-whisper` checkpoint on a
rented GPU; re-benchmark through the harness. *1–2 weeks human transcription + ~1 day GPU, ~$50
compute.* **Expected: CER 8–15 %.**

### 5.6 Cross-cutting — the keyterm gazetteer

A **Somali food-security gazetteer** (~500 terms — place names plus *abaar*/drought,
*macluul*/famine, *biyo*/water, *suuq*/market, *qaxooti*/refugee …). Used twice: keyterm-F1 scoring
in Initiative A, and ASR prompt-boosting where the engine supports it. It also serves the topic
stage, where English `en_core_web_sm` NER currently runs over translated text to find Somali place
names — recall is likely poor and is unmeasured.

New optional dependencies stay in extras and are lazy-imported per repo convention: `jiwer`
(metrics), `silero-vad` / `pyannote.audio` (VAD), plus engine-specific packages. All new model ids
and thresholds go in `config/config.yaml`.

---

## 6. Sequencing

| Phase | Work | Exit gate | Effort |
|---|---|---|---|
| **0 — Stop being wrong** | §3: bugs, dead config, seeds, uv/ruff/CI, merge to main | Config is live, themes are stable, tests run on push | ~2–3 days ✅ |
| **0.5 — Re-collect** | Rebuild `data/`: broadcasts, IPC releases, feedback PDFs | A week's slice runs end to end through 03 → 04 → 06 | ~1–2 days |
| **1 — Does this work?** | §4: IPC validation + signal labelling | A number for "beats no-change: yes / no" | ~3–4 days |
| **2a — *if yes*: amplify** | §5 A→D, plus translation and topic metrics | Best engine shipped, every stage has a metric | ~2–3 weeks |
| **2b — *if no*: rebuild the rule** | Replace keyword+threshold with a fitted model on labelled signals | Beats no-change, or the finding is written up honestly | ~1–2 weeks |
| **3 — conditional** | §5.5 fine-tune | Only if no off-the-shelf model clears the decision rule | ~2 weeks |

Phases 0.5 and 1 are about a week together, and they determine whether Phase 2's ~$250 and several
weeks are the right next investment. That is the whole argument for running them first.

The one exception to strict ordering: because the archive is being re-collected anyway, the
Initiative B benchmark should run *during* Phase 0.5, so the bulk transcription uses the winning
engine.

---

## 7. Open questions

- **Is there historical IPC ground truth aligned with the feedback-PDF period?** Phase 1 depends
  on it entirely. If the PDFs and the IPC releases do not overlap enough, the validation needs a
  different design — worth checking before anything else on this list.
- **A negative result is a publishable finding here.** If radio feedback does not improve on the
  baseline, that is worth writing up for Zero Hunger Lab. Is the project set up to report that, or
  is a positive result assumed?
- **Compute access** — is a GPU (A100 / L40S, owned or rented) available for self-hosting, or
  should the plan lean on managed APIs? This decides the feasibility and cost of Initiatives B and
  E (~$50 self-host vs ~$1,300 for Chirp 3 across the full ~900 h archive).
- **Gold-set transcription** — hire a native Somali transcriber (~$200), or use Radio Ergo's own
  correspondent network?
- **Data residency** — if GDPR / EU residency applies to the Zero Hunger Lab collaboration,
  self-hosting or EU-region API endpoints are the cleanest path.
- **Does the SoundCloud collector still work?** It has not been run since the archive was lost;
  page markup changes silently.
- **Transcribe + translate coupling** — SeamlessM4T v2 can do Somali→English speech translation in
  one step (the only strong open option). If of interest, this could eventually merge the
  transcribe and translate stages.

---

## Appendix A — ASR evidence base

> Research compiled May 2026, preserved as the sourced material behind §5. Every Somali number
> below is a **starting hypothesis to confirm on our own gold set**, not a result.

### A.1 Key findings

**1. Whisper large-v3 effectively does not work zero-shot on Somali.** OpenAI's own
`language-breakdown.svg` excludes Somali because Whisper large-v3 exceeds 60 % error rate on it
(OpenAI/whisper GitHub discussions #1762, #234). The Fleurs-SLU paper (Schmidt et al., COLM 2025;
arXiv:2501.06117v3, Table 8) measured Whisper-v3-Large at **34.9 % CER (σ=28.7)** on FLEURS Somali
(`som_Latn`), with Som→Eng speech translation collapsing to **0.4 sacreBLEU** — unusable. A common
failure mode is *script collapse*: Whisper outputs Arabic/Sinhala script for Somali audio.

**2. Meta SeamlessM4T v2 Large is the strongest open-weights zero-shot option for Somali.** Same
table reports **17.9 % CER (σ=18.8)** and **18.1 BLEU Som→Eng** — roughly half the error of Whisper
and 45× the speech-translation BLEU. Officially supports Somali for ASR and S2TT
(`facebook/seamless-m4t-v2-large`). Chunk-tolerant for long audio when wrapped with VAD.

**3. Meta MMS supports Somali but the per-language number is buried.** The MMS paper (Pratap et
al., JMLR 2024, vol 25) reports aggregate FLEURS-102 CER of ~6.2–6.3 % for the multi-domain MMS-1B
model; the Somali-specific number sits in Appendix D and could not be independently verified.
Supports Latin-script Somali natively via the `som` adapter; runs on a 16 GB GPU.

**4. Meta Omnilingual ASR (`facebook/omniASR-LLM-7B`, Nov 2025) is a promising entrant.** Per
arXiv:2511.09690, verbatim: *"Our 7B-LLM-ASR system achieves state-of-the-art performance across
1,600+ languages, with character error rates (CER) below 10 for 78 % of those languages."* Somali
is in the supported list; the precise Somali row was not extractable. Benchmark before trusting.

**5. Managed APIs that officially support Somali:**
- **Google Cloud Speech-to-Text Chirp 2 / Chirp 3 (BatchRecognize)** — supports `so-SO`.
- **Amazon Transcribe** — added Somali Oct 2024. The announcement covers *streaming*; batch support
  should be confirmed in current docs before relying on it.
- **AssemblyAI Universal-2 fallback** — Somali (`so`) in the 99-language tier.
- **ElevenLabs Scribe v1 / v2** — Somali supported but assigned to the *"Moderate (>20 % to ≤50 %
  WER)"* tier. The "3.1 % on FLEURS" claim on the Somali landing page is boilerplate that appears
  identically on every Scribe language page and is **not** a Somali-specific measurement.
- **OpenAI Whisper API** — Somali nominally listed, in practice high WER (see #1).

**6. Major commercial APIs that do NOT support Somali:** Microsoft Azure AI Speech (Somali is TTS
only), Deepgram Nova-3 / Nova-2 / Flux, Speechmatics, NVIDIA Canary-1B-v2 / Parakeet-TDT-0.6B-v3
(25 European languages only, arXiv:2509.14128), AssemblyAI Slam-1 / Universal-3 Pro.

**7. Hugging Face community fine-tunes for Somali:**
- `steja/whisper-large-somali` — **WER 55.04 %** on FLEURS so_so (model card).
- `steja/whisper-small-somali` — **WER 66.60 %** on FLEURS so_so.
- `Mustafaa4a/ASR-Somali` (Wav2Vec2-XLSR-53) — **WER 30.60 %** on its own held-out set, not FLEURS.
  *This is the current production model.*
- `laalays/mms1b-finetuned-somali-2` — WER 71.71 %; prototype, not production-grade.
- `microsoft/paza-whisper-large-v3-turbo` — fine-tuned on 6 Kenyan languages including Somali.
  The most relevant recent baseline; try it before fine-tuning our own.

**8. Pricing (USD, May 2026).** ElevenLabs Scribe v2 batch $0.22–0.40/hr · AssemblyAI Universal-2
$0.27/hr · OpenAI Whisper API $0.36/hr · Google Chirp 2/3 ~$1.44/hr (batch + GCS discounts) · AWS
Transcribe ~$1.44/hr tiered · Deepgram Whisper Cloud ~$0.29/hr (but Somali quality is Whisper's) ·
**self-hosted on a rented L40S/A10 (~$0.50–0.80/wall-clock-hour) transcribes 10–30 audio-hours per
wall-clock hour → ~$0.02–0.08 per audio-hour.** For the ~900 h archive that is $20–50 self-hosted
vs ~$1,300 managed.

**9. Public Somali training data is thin but enough for a useful fine-tune.** FLEURS Somali ~12 h
read speech · Common Voice 20 Somali (no per-language breakdown published; low single-digit
validated hours — verify on download) · ALFFA does *not* include Somali · **Stellenbosch / UN Global
Pulse Somali: 1.57 h seed + 17.55 h semi-supervised radio audio (Biswas et al. 2019,
arXiv:1907.03064) — exactly our domain, the original UN radio-browsing pilot for Somali
humanitarian monitoring; worth contacting Niesler's group directly** · CS224S Stanford Somali ·
**Radio Ergo's own archive is by far the largest leverage.**

**10. Robustness factors specific to Radio Ergo.** Long files (25–30 min) need VAD chunking for
Whisper and SeamlessM4T; Chirp 3 BatchRecognize is the most set-and-forget managed path. Telephone
audio: Deepgram's analysis found Whisper-v3 median WER on phone calls is **42.9 %** even for
languages it handles well — assume Somali phone-call WER is materially worse. Music stings:
`pyannote` or `silero-vad` typically removes 5–15 % of garbage. Code-switching
(Somali↔Arabic↔English) is pervasive in Radio Ergo and unmeasured by every public benchmark;
SeamlessM4T v2 is the most code-switch tolerant open option.

### A.2 Comparison table

| Option | Somali official? | Best evidence (Somali) | Deployment | Cost / hr | Long-file | Diarization | Notes for radio |
|---|---|---|---|---|---|---|---|
| **SeamlessM4T v2 Large** (Meta, open) | Yes (ASR + S2TT) | **17.9 % CER FLEURS** | Self-host, A100 80GB | ~$0.05–0.10 | Needs VAD | No | Strongest zero-shot; needs noise-robust eval |
| **OmniASR-LLM-7B** (Meta, Nov 2025) | Yes (1,600+ langs) | "CER below 10 for 78 % of languages"; Somali row not extracted | Self-host, ~24 GB | ~$0.05–0.10 | Yes | No | Benchmark before committing |
| **paza-whisper-large-v3-turbo** (MSR) | Yes (Somali fine-tune) | CER/WER charts vs OmniASR-7B, MMS-1B, Whisper-v3-turbo | Self-host, ~10 GB | ~$0.02–0.05 | Yes | No | Probably the best ready-to-run open model |
| **Whisper large-v3 / turbo** | Nominal, ≥60 % error | 34.9 % CER FLEURS; wrong-script output | Self-host / API | $0.36 API | Yes | No | Only useful after fine-tuning |
| **MMS-1B-all** (Meta) | Yes (`som` adapter) | Aggregate FLEURS-102 CER ~6.2 %; Somali unverified | Self-host, 16 GB | ~$0.02–0.05 | No (use VAD) | No | Lightweight; reliably Latin script |
| **Mustafaa4a/ASR-Somali** | Yes | WER 30.6 % on own test | Self-host | low | No | No | **Current production model**; not radio-tuned |
| **Google Chirp 2 / 3 Batch** | Yes | No public per-Somali WER | Managed | ~$1.44 | Yes | Yes | Easiest "just works" path |
| **Amazon Transcribe** | Streaming confirmed; verify batch | None published | Managed | ~$1.44 | Yes | Yes | No published Somali quality data |
| **ElevenLabs Scribe v2** | "Supported", Moderate tier | The 3.1 % claim is not Somali-specific | Managed | $0.22–0.40 | ≤2 hr/file | Yes (48 spk) | Don't trust the landing page |
| **AssemblyAI Universal-2** | Yes (tier 5) | None published | Managed | $0.27 | Yes | Yes | Somali quality unverified |
| **Deepgram (Whisper Cloud)** | Indirect | Inherits Whisper's poor Somali | Managed | $0.29 | ≤20 min | No | Not recommended |

### A.3 Evaluation-design notes

**Normalisation pitfalls, Somali-specific.** Lowercase and strip punctuation, but **keep the
apostrophe** if the tokeniser uses it for the glottal stop — normalise reference and hypothesis
identically. Strip non-Latin characters *before* scoring or Whisper's script collapse pushes CER
past 100 %; track script fidelity as its own metric. Decide one convention for numerals (models
emit "2025" where speakers say *labba kun iyo shan iyo labaatan*) and apply it to both sides. Keep
code-switched English/Arabic words as-is in both.

**Why CER over WER.** Somali's agglutinative compounds and morphological inflection drive WER up
without changing semantic content (Thennal & Gopinath 2024, arXiv:2410.07400; Kuhn et al. 2024,
arXiv:2408.15616 on normalisation pitfalls).

**Keyterm F1 matters more than aggregate WER** for this pipeline: tag IPC-critical entities in the
reference (place names, *abaar* / *macluul* / *biyo* / *suuq* / *qaxooti*) and score precision and
recall of those entities in the hypothesis.

### A.4 Caveats

- **Clean-read benchmarks understate real error.** FLEURS and Common Voice are studio read speech.
  Expect any FLEURS number here to understate Radio Ergo error by **5–20 absolute CER points**.
- **Several published Somali numbers cannot be independently verified** — MMS Appendix D, Google
  USM aggregate-only, OmniASR's implied Somali row, and no Somali-specific WER at all from
  ElevenLabs or AssemblyAI.
- **Vendor marketing needs verification.** ElevenLabs' "3.1 % WER on FLEURS" is repeated across 99
  language pages; their own tiering places Somali in the 20–50 % bucket.
- **OmniASR's comparative claim is self-reported**, and is a relative reduction, not absolute points.
- **AWS Transcribe Somali batch support needs confirming** — the Oct 2024 announcement covers
  streaming.
- **Code-switching is under-evaluated.** CS-FLEURS (arXiv:2509.14161) covers 52 languages; Somali is
  not one. We would have to evaluate this ourselves.
- **Dialect coverage is unknown.** All published numbers are Northern (Standard) Somali — the
  broadcast dialect. Maay (~25 % of speakers) is essentially uncovered; expect worse quality from
  Maay-speaking field correspondents.
- **Whisper script-collapse is a hidden failure mode** — Arabic-script gibberish looks plausible but
  is garbage downstream. Always check the output is Latin script.

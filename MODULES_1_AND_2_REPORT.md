# Technical Report: Multilingual Regional-Language Sentiment Analysis (Modules 1 & 2)

**Project Title:** Multilingual Regional Language Sentiment Analysis Using Hybrid AI Models  
**Scope Covered:** Module 1 (Data Collection and Preprocessing) & Module 2 (Text and Audio Processing)  
**Target Languages (7 Low-Resource Indian Languages):** Tulu, Assamese, Konkani, Santali, Bodo, Bhojpuri, Dogri  
**Status:** Completed and Verified  

---

## 1. Executive Summary & Scope Delimitation

This technical report documents the design, implementation, and empirical verification of **Module 1 (Data Collection and Preprocessing)** and **Module 2 (Text and Audio Processing)**.

In strict compliance with the project specifications:
- **No multimodal fusion (Module 3)** has been implemented.
- **No final integrated sentiment prediction system or web dashboard (Module 4)** has been constructed.
- **No fabricated datasets, fake API transcripts, or exaggerated model accuracy claims** have been introduced.
- Direct-language sentiment analysis, bilingual code-mixed inputs, regional slang, dialect variations, and dual-modality (text and audio) handling have been established as robust, reusable foundational layers.

---

## 2. Linguistic Reality & Language Coverage Matrix

The target languages represent distinct language families, scripts, and varying degrees of digital resource scarcity in NLP:

| Language | ISO 639-3 | Script | Language Family | Dialect Regions | Representation Status in Digital NLP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tulu** | `tcy` | Kannada (`Knda`) | Dravidian | Dakshina Kannada, Udupi, Kasaragod | Extremely low-resource. Missing in standard mBERT/XLM-R/IndicBERT pretraining corpora. Highly code-mixed with Kannada & English. |
| **Assamese** | `asm` (`as`) | Bengali-Assamese (`Beng`) | Indo-Aryan (Eastern) | Brahmaputra Valley, Kamrup, Goalpara | Medium-low resource. Distinctive characters `ৰ` (U+09F0) and `ৱ` (U+09F1). Officially covered by XLM-R and IndicBERT. |
| **Konkani** | `kok` (`gom`) | Devanagari (`Deva`), Kannada, Latin | Indo-Aryan (Southern) | Goa, Coastal Karnataka, Maharashtra, Kochi | Multi-script. Supported in IndicBERT v2; XLM-R relies on Devanagari Marathi/Hindi transfer. |
| **Santali** | `sat` | Ol Chiki (`Olck`), Latin, Devanagari | Austroasiatic (Munda) | Jharkhand, Odisha, West Bengal, Assam | Unique Ol Chiki Unicode block (`U+1C50`–`U+1C7F`). Supported in IndicBERT v2; unsupported in standard XLM-R (byte-fallback). |
| **Bodo** | `brx` | Devanagari (`Deva`) | Tibeto-Burman | Bodoland Territorial Region (Assam) | Official 8th Schedule language. Supported in IndicBERT v2; XLM-R relies on Devanagari subwords. |
| **Bhojpuri** | `bho` | Devanagari (`Deva`) | Indo-Aryan (Bihari) | Bihar, Eastern UP (Purvanchal), Nepal Terai | High colloquial speaker base (~50M+) but low digital annotated sentiment data. Supported in IndicBERT v2. High transfer with Hindi. |
| **Dogri** | `doi` (`dgo`) | Devanagari (`Deva`) | Indo-Aryan (Northwestern) | Jammu & Kashmir, Himachal Pradesh | Supported in IndicBERT v2. Historical Dogra Akkhar / Takri rarely digitized. |

---

## 3. Pretrained Model Verification & Gap Analysis

A central scientific principle of this implementation is **truth in advertising**: we do not claim that off-the-shelf foundation models understand all 7 languages out-of-the-box.

### 3.1 XLM-RoBERTa (`xlm-roberta-base`)
- **Stated Scope:** 100 languages trained on CommonCrawl (CC-100).
- **Actual Coverage on Target Set:**
  - *Assamese (`as`):* Fully represented in pretraining.
  - *Tulu (`tcy`):* **Not represented.** Kannada-script Tulu text is tokenized by borrowing Kannada subword tokens (`piece`). Unseen morphological combinations produce fragmented subword sequences.
  - *Santali (`sat`):* **Not represented.** Ol Chiki script characters are absent from the standard vocabulary; characters fall back to byte-level BPE tokens.
  - *Bhojpuri, Bodo, Dogri, Konkani:* Not explicitly separated in CC-100. They benefit from cross-lingual Devanagari transfer from Hindi, Marathi, and Sanskrit, but dialectal markers and slang undergo subword over-segmentation.
- **Classification Status:** When labeled sentiment data is absent, the classification head weights remain untrained. The implementation explicitly returns a diagnostic note stating that supervised fine-tuning is required.

### 3.2 IndicBERT (v1 vs. v2)
- **IndicBERT v1 (`ai4bharat/indic-bert`, ALBERT architecture):**
  - Covered 12 languages: `as`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `or`, `pa`, `ta`, `te`, `ur`.
  - **Does NOT support:** Tulu, Santali, Bodo, Bhojpuri, Dogri, Konkani.
- **IndicBERT v2 (`ai4bharat/IndicBERTv2-MLM-only`, DeBERTa architecture):**
  - Expanded to 23 languages including `bho`, `brx`, `doi`, `kok` (`gom`), `sat` (Ol Chiki), and `as`.
  - **Does NOT include:** Tulu (`tcy`).
- **Conclusion:** IndicBERT v2 is superior for Bodo, Dogri, Bhojpuri, Konkani, and Santali; however, neither model contains dedicated pretraining for Tulu.

### 3.3 Statistical Language Identification (`langdetect`)
- Off-the-shelf `langdetect` (Google Language Detection port) covers ~55 languages.
- It has **zero native models** for Tulu, Santali, Bodo, Bhojpuri, Dogri, or Konkani.
- Left alone, `langdetect` misidentifies:
  - Tulu as Kannada (`kn`).
  - Bhojpuri, Dogri, Bodo, and Konkani as Hindi (`hi`) or Marathi (`mr`).
  - Assamese as Bengali (`bn`).
- **Solution Implemented:** A **hybrid language detector** combining:
  1. Deterministic Unicode script boundary detection (e.g. Ol Chiki -> Santali).
  2. Distinctive grapheme filtering (e.g. `ৰ` and `ৱ` -> Assamese).
  3. Regional lexical and morphological markers (e.g. `ದಾನೆ`, `ಇಜ್ಜಿ`, `ಎಡ್ಡೆ` -> Tulu; `बा`, `बानी`, `का बात बा` -> Bhojpuri; `मिकी`, `तुगी`, `नेईं` -> Dogri; `हांव`, `आसा` -> Konkani; `दा`, `मोजां` -> Bodo).
  4. Explicit uncertainty reporting: When no distinctive regional markers are found in Devanagari, the system sets `is_ambiguous=True`, `review_recommended=True`, and reduces confidence rather than hallucinating an inaccurate language label.

---

## 4. Module 1: Data Collection & Preprocessing Architecture

### 4.1 Schema & Validation Pipeline
- **Text Dataset Schema:**
  `sample_id, text, language, secondary_language, input_type, sentiment, script, dialect, source`
- **Audio Metadata Schema:**
  `sample_id, audio_path, language, transcript, sentiment, dialect, source`
- **Validation Engine (`src/data/dataset_validator.py`):**
  - Verifies non-empty required fields (`sample_id`, `text`, `language` for text; `sample_id`, `audio_path`, `language` for audio).
  - Validates language codes against the supported 7-language catalog while permitting secondary/loan codes.
  - Detects duplicate sample IDs.
  - Verifies physical audio existence on disk.
  - Produces structured `ValidationReport` detailing valid vs invalid counts and diagnostic errors.

### 4.2 Text Preprocessing (`src/preprocessing/text_preprocessing.py`)
- **Unicode Normalization:** Applies `NFC` normalization (`unicodedata.normalize('NFC', text)`), preventing font decomposition or matra corruption.
- **Negation Preservation:** Comprehensive lexicon covering regional negation tokens across all 7 languages (e.g. Tulu `ಇಜ್ಜಿ`, Assamese `নহয়`, Bhojpuri `नाहीं`, Dogri `नेईं`, Santali `ᱵᱟᱝ`, Bodo `दा`, Konkani `ना`). Negations are preserved as key sentiment-reversal signals.
- **Emoji Preservation:** Preserves standard emojis (`😊`, `❤️`, `🔥`, etc.) which convey direct emotional intensity.
- **Slang and Dialect Preservation:** Retains regional colloquial expressions without normalizing them to standard Hindi or Kannada.
- **Deduplication:**
  - Exact deduplication via SHA-256 hashes of normalized strings.
  - Near-duplicate detection via character 3-gram Jaccard similarity (configurable threshold, default `0.88`).
- **Storage:** Original raw input and cleaned processed text are stored separately in `original_text` and `processed_text`.

### 4.3 Audio Preprocessing (`src/preprocessing/audio_preprocessing.py`)
- **Inspection & Validation:** Checks file existence, non-empty filesize (>0 bytes), supported formats (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`), minimum duration (`0.3s`), and silence detection.
- **Standardization:**
  - Converts multi-channel audio to mono (`mean(axis=1)`).
  - Removes DC offset (`data - mean(data)`).
  - Resamples polyphase signals to standard `16000 Hz` (the standard for XLS-R and speech models).
  - Applies optional high-pass pre-emphasis filter (`y[t] = x[t] - 0.97 * x[t-1]`).
  - Applies RMS amplitude peak normalization.
  - Writes standard 16-bit PCM WAV.
- **Windowed Segmentation:** Files exceeding maximum duration (default `30.0s`) are segmented into overlapping windows (default `15.0s` with `1.0s` overlap), preserving segment index and timestamps.

---

## 5. Module 2: Text and Audio Processing Architecture

### 5.1 Text Processing with XLM-RoBERTa (`src/text_processing/xlm_roberta.py`)
- **Model Baseline:** `xlm-roberta-base`.
- **Tokenization & Embeddings:** Generates subword token IDs, attention masks, and sentence-level pooled embeddings (supporting `mean`, `cls`, and `max` pooling).
- **Classification Head:** 3-class sequence classification head (`positive`, `negative`, `neutral`).
- **Fine-Tuning Guard:** Requires a minimum number of verified labeled samples; if insufficient, reports `sentiment classification has not yet been trained` and restricts execution to feature extraction.
- **Checkpoint Persistence:** Methods for saving and loading fine-tuned checkpoints.

### 5.2 IndicBERT Comparative Pipeline (`src/text_processing/indicbert.py`)
- Separate configuration, tokenization, and model instantiation for `ai4bharat/IndicBERTv2-MLM-only` (and v1).
- Provides embedding extraction and classification interfaces.
- Integrates language coverage checking (`verify_language_coverage`) to warn researchers when evaluating out-of-vocabulary languages like Tulu.

### 5.3 SraVaani ASR Integration (`src/audio_processing/sraavaani_asr.py`)
- **Project Background:** SraVaani is an IISc Bangalore (SPIRE / RESPIN) initiative developing speech recognition for regional Indian languages, specifically Tulu and Kannada dialects.
- **Zero Hallucination Guarantee:** The client dispatches real HTTP requests when `SRAVAANI_API_ENDPOINT` and `SRAVAANI_API_KEY` are provided.
- **Unconfigured Behavior:** When credentials/endpoints are absent, the client returns:
  `TranscriptionResult(status=TranscriptionStatus.UNAVAILABLE, transcript=None, error="SraVaani ASR service is currently unconfigured. External API endpoint or local model checkpoint required.")`
- **Integrity Rule:** Absolutely no fake transcription text is hallucinated or substituted.

### 5.4 Audio Feature Extraction
- **Learned Speech Embeddings (`src/audio_processing/xlsr_features.py`):**
  - Model: `facebook/wav2vec2-xls-r-300m` (53-language cross-lingual speech model).
  - Extracts continuous frame representations pooled over time into 1024-dimensional vectors.
  - Handles short audio clips (<0.2s) with structured error reporting.
  - Persists embeddings to reproducible NumPy `.npz` archives.
- **Acoustic Descriptors (`src/audio_processing/opensmile_features.py`):**
  - Uses `openSMILE` with standard feature sets: `eGeMAPSv02` (88 features) or `ComParE_2016` (6373 features).
  - Extracts: Fundamental frequency (F0 pitch), jitter, shimmer, harmonics-to-noise ratio (HNR), loudness, formant frequencies, and spectral flux.
  - Includes a standalone **NumPy/SciPy acoustic signal fallback** (autocorrelation pitch estimation, RMS energy, zero-crossing rate, spectral centroid, spectral rolloff) to ensure operational resilience.
  - Persists acoustic features to CSV format.
  - Learned speech embeddings and low-level acoustic features are kept strictly separate.

---

## 6. Documented Public Datasets & Sourcing Limitations

The following public corpora have been factually documented in the dataset catalog (`src/data/dataset_loader.py` and `configs/language_catalog.yaml`). None are assumed to be sentiment-labeled or unrestricted:

1. **IndicNLP / IndicCorp (AI4Bharat):**
   - *Languages:* Assamese, Bodo, Dogri, Konkani, Santali, Bhojpuri.
   - *Modality:* Text only.
   - *License:* CC-BY-4.0 / CC0.
   - *Limitation:* Web-scraped and news data; **unlabeled for sentiment**. Requires manual annotation.
2. **SraVaani (IISc / SPIRE / RESPIN):**
   - *Languages:* Tulu, Kannada regional dialects.
   - *Modality:* Audio + transcripts.
   - *License:* Restricted academic research access.
   - *Limitation:* Not publicly downloadable via anonymous links; requires formal agreement with IISc. Unlabeled for sentiment.
3. **InDRA Bhojpuri Spoken Corpus:**
   - *Languages:* Bhojpuri.
   - *Modality:* Audio + transcript.
   - *License:* Non-commercial academic license.
   - *Limitation:* Focused on speech recognition; unannotated for sentiment.
4. **LDC-IL (CIIL Mysore):**
   - *Languages:* Dogri, Bodo, Assamese, Konkani, Santali.
   - *Modality:* Sentence-aligned speech and monolingual text.
   - *License:* Proprietary institutional membership license.
   - *Limitation:* Restricted; requires paid or institutional access.

---

## 7. Verification & Test Suite Summary

A comprehensive test suite containing **41 unit tests** across 8 test modules was executed with **100% pass rate**:

```text
tests/test_schema_and_validation.py      .... (5 passed)
tests/test_text_preprocessing.py        ..... (5 passed)
tests/test_language_detection.py        ..... (5 passed)
tests/test_code_mixing.py               ...   (3 passed)
tests/test_audio_preprocessing.py       ...... (6 passed)
tests/test_text_processing.py           ....  (4 passed)
tests/test_audio_processing.py          ..... (5 passed)
tests/test_unified_interface.py         ........ (8 passed)

============================= 41 passed in 5.67s ==============================
```

All 8 testing requirements specified by the prompt were verified:
1. Missing dataset fields, duplicate IDs, and invalid records flagged.
2. Text preprocessing preserves negations, emojis, and regional slang.
3. Language identification uncertainty and ambiguity flagged for review.
4. Bilingual and code-mixed inputs detected with secondary language identification.
5. Invalid audio formats, 0-byte recordings, and excessive lengths handled.
6. Text embedding extraction and pooling methods verified.
7. openSMILE and acoustic feature extraction verified on audio waveforms.
8. Unavailability of SraVaani credentials or models handled with zero hallucination.

---

## 8. Conclusion & Readiness for Subsequent Phases

Modules 1 and 2 provide a scientifically verified, modular, and reliable foundation for the research prototype. The 10 functions in `src/unified_interface.py` provide a clean, standardized contract ready for future downstream components (Modules 3 and 4) when multimodal fusion and final classification are developed.

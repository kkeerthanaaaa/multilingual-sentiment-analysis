# Multilingual Regional-Language Sentiment Analysis Using Hybrid AI Models

## Research Prototype: Modules 1 & 2

This repository implements **Module 1 (Data Collection and Preprocessing)** and **Module 2 (Text and Audio Processing)** of the research project **"Multilingual Regional Language Sentiment Analysis Using Hybrid AI Models"**.

The project focuses on seven low-resource Indian languages:
1. **Tulu (`tcy`)** (Kannada script)
2. **Assamese (`as` / `asm`)** (Bengali-Assamese script)
3. **Konkani (`kok` / `gom`)** (Devanagari, Kannada, Romi/Latin scripts)
4. **Santali (`sat`)** (Ol Chiki, Latin, Devanagari scripts)
5. **Bodo (`brx`)** (Devanagari script)
6. **Bhojpuri (`bho`)** (Devanagari script)
7. **Dogri (`doi` / `dgo`)** (Devanagari script)

---

## Architecture Overview

```text
multilingual-sentiment/
├── data/
│   ├── raw/
│   │   ├── text/                 # Raw CSV/JSONL datasets & manual guide
│   │   └── audio/                # Raw audio files and metadata CSV
│   ├── processed/                # Standardized 16kHz audio & preprocessed text
│   └── metadata/                 # Metadata records
├── src/
│   ├── unified_interface.py      # Top-level reusable API (10 unified functions)
│   ├── data/
│   │   ├── schema.py             # Pydantic schemas (TextRecord, AudioRecord)
│   │   ├── dataset_validator.py  # Schema validation & duplicate detection
│   │   └── dataset_loader.py     # CSV/JSONL loader & public resource catalog
│   ├── preprocessing/
│   │   ├── text_preprocessing.py # Script, negation, emoji & slang preservation
│   │   ├── language_detection.py # Hybrid script/marker language identifier
│   │   └── audio_preprocessing.py# 16kHz conversion, validation, segmentation
│   ├── text_processing/
│   │   ├── xlm_roberta.py        # XLM-RoBERTa encoder & classification head
│   │   └── indicbert.py          # IndicBERT v1/v2 comparative pipeline
│   ├── audio_processing/
│   │   ├── sraavaani_asr.py      # SraVaani ASR client (zero-hallucination)
│   │   ├── xlsr_features.py      # XLS-R learned speech representations
│   │   └── opensmile_features.py # openSMILE eGeMAPSv02 & acoustic descriptors
│   └── utils/
│       ├── configuration.py      # YAML config loader
│       └── logging_config.py     # Structured logging
├── configs/
│   ├── default_config.yaml       # Default pipeline parameters
│   └── language_catalog.yaml     # Detailed linguistic & model coverage catalog
├── tests/                        # 41 unit tests covering validation, NLP, audio
├── requirements.txt              # Production dependency specifications
├── pytest.ini                    # Test runner configuration
├── main.py                       # CLI demo and validation tool
└── MODULES_1_AND_2_REPORT.md     # Comprehensive research & limitations report
```

---

## Installation

### Prerequisites
- Python 3.11
- Windows, Linux, or macOS

### Setup Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Quickstart & Demonstration

Fetch real-time multilingual datasets (from AI4Bharat IndicSentiment & regional corpora):
```bash
python main.py --fetch-real-data
```

Train sentiment model on real-time dataset:
```bash
python main.py --train --model-type hybrid
```

Run real-time sentiment prediction on regional text:
```bash
python main.py --predict-text "ದಾನೆ ಮಾರ್ರೆ ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು!"
```

Run the automated pipeline demonstration on synthetic test fixtures:
```bash
python main.py
```

Validate a custom text dataset:
```bash
python main.py --validate-text data/raw/text/real_time_multilingual_sentiment_dataset.csv
```

Inspect an audio file:
```bash
python main.py --validate-audio data/raw/audio/synthetic_tcy_sample.wav
```

Run test suite:
```bash
pytest
```

---

## Unified Interface (10 Core Functions)

Import directly from `src.unified_interface`:

```python
from src.unified_interface import (
    load_text_dataset,
    preprocess_text,
    identify_language,
    detect_code_mixing,
    validate_audio,
    preprocess_audio,
    transcribe_audio,
    extract_text_embeddings,
    extract_audio_embeddings,
    extract_acoustic_features,
)

# 1. Dataset loading & validation
df, records, report = load_text_dataset("data/raw/text/synthetic_sample_text_dataset.csv")

# 2. Text preprocessing (preserves scripts, negations, emojis, slang)
res = preprocess_text("ದಾನೆ ಮಾರ್ರೆ ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು! 😊")
print(res.processed_text, res.detected_emojis)

# 3. Language identification & ambiguity flagging
lang_res = identify_language("ᱱᱚᱣᱟ ᱯᱷᱤᱞᱢ ᱟᱹᱰᱤ ᱵᱷᱟᱹᱜᱤ ᱜᱮᱭᱟ!")
print(lang_res.primary_language, lang_res.confidence, lang_res.primary_script)

# 4. Code-mixing detection
is_mixed, ratio, sec_lang = detect_code_mixing("Movie was bad, ಎಂಕ್ ಇಷ್ಟ ಇಜ್ಜಿ at all!")

# 5. Audio inspection
inspection = validate_audio("data/raw/audio/synthetic_tcy_sample.wav")

# 6. Audio standardization (resampling to 16kHz mono, segmentation)
audio_res = preprocess_audio("data/raw/audio/synthetic_tcy_sample.wav")

# 7. SraVaani ASR transcription (reports unavailable if unconfigured; no hallucinated output)
asr_res = transcribe_audio("data/raw/audio/synthetic_tcy_sample.wav", language="tulu")

# 8. Text embedding extraction (XLM-RoBERTa or IndicBERT)
embeddings = extract_text_embeddings(["का बात बा भाई!", "बोरें आसा"], model_type="xlm-roberta")

# 9. Learned speech embeddings (XLS-R)
xlsr_out = extract_audio_embeddings(["data/raw/audio/synthetic_tcy_sample.wav"])

# 10. Acoustic feature extraction (openSMILE eGeMAPSv02)
acoustic_out = extract_acoustic_features(["data/raw/audio/synthetic_tcy_sample.wav"])
```

---

## Scientific Integrity & Scope Compliance

1. **Modules 1 & 2 Only**: Does not implement multimodal fusion or final sentiment classification (reserved for Modules 3 & 4).
2. **Zero Hallucination Policy**: SraVaani ASR returns explicit `TranscriptionStatus.UNAVAILABLE` when external credentials or endpoints are unconfigured. No fake words are fabricated.
3. **Factual Language Coverage**: Honest documentation of model limitations. XLM-R and IndicBERT do not natively support all 7 languages equally. Tulu relies on Kannada subword tokenization, and Santali Ol Chiki falls back to subwords/bytes in XLM-R.
4. **Synthetic Fixtures**: All unit test files and samples in `data/raw/` are explicitly marked as synthetic fixtures.

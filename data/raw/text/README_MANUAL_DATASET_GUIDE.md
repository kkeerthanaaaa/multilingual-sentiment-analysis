# Manual Dataset Ingestion Guide for Low-Resource Indian Languages

This guide details how researchers and practitioners can manually add custom text or audio datasets to this pipeline.

> **CRITICAL SCIENTIFIC INTEGRITY NOTE:**
> In accordance with research integrity standards, do not fabricate download links, dataset labels, or contents. When sourcing third-party data, verify the explicit license and annotate whether sentiment labels were human-verified, pseudo-labeled, or unannotated.

---

## 1. Text Datasets

Place your raw text dataset in `data/raw/text/` as either a `.csv` or `.jsonl` file.

### Required and Optional CSV Header / JSON Schema

```text
sample_id,text,language,secondary_language,input_type,sentiment,script,dialect,source
```

| Field Name | Type | Required? | Allowed Values / Examples | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `sample_id` | String | **Yes** | e.g., `tulu_001`, `bho_042` | Must be non-empty and unique per dataset. |
| `text` | String | **Yes** | e.g., `ದಾನೆ ಮಾರ್ರೆ ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು!` | Raw original text before preprocessing. |
| `language` | String | **Yes** | `tulu`, `assamese`, `konkani`, `santali`, `bodo`, `bhojpuri`, `dogri` (or ISO `tcy`, `as`, `kok`, `sat`, `brx`, `bho`, `doi`) | Primary target language. |
| `secondary_language` | String | No | `en`, `hi`, `kn`, etc. | Present if bilingual or code-mixed. |
| `input_type` | String | **Yes** | `direct`, `bilingual`, `code_mixed` | Defaults to `direct` if unstated. |
| `sentiment` | String | No | `positive`, `negative`, `neutral`, `unlabeled` | Leave blank or set to `unlabeled` if unannotated. |
| `script` | String | No | `Devanagari`, `Kannada`, `Bengali`, `Ol Chiki`, `Latin` | Script used in text. |
| `dialect` | String | No | e.g., `Brahmin Tulu`, `Kamrupi`, `Gorakhpuri` | Regional dialect variation. |
| `source` | String | No | e.g., `IndicNLP`, `SraVaani`, `Fieldwork` | Dataset provenance. |

### Example JSONL Line

```json
{"sample_id": "bho_001", "text": "ई सिनेमा एकदम नीक बा!", "language": "bhojpuri", "secondary_language": null, "input_type": "direct", "sentiment": "positive", "script": "Devanagari", "dialect": "Bhojpuriya", "source": "Manual_Survey"}
```

---

## 2. Audio Datasets

Store raw audio recordings in `data/raw/audio/` (e.g., `data/raw/audio/tulu_recordings/`).
Create a metadata file in `data/metadata/` or alongside the audio in CSV or JSONL format.

### Audio Metadata Schema

```text
sample_id,audio_path,language,transcript,sentiment,dialect,source
```

| Field Name | Type | Required? | Example |
| :--- | :--- | :--- | :--- |
| `sample_id` | String | **Yes** | `aud_tcy_001` |
| `audio_path` | String | **Yes** | `data/raw/audio/tulu_01.wav` (relative or absolute) |
| `language` | String | **Yes** | `tulu` |
| `transcript` | String | No | `ಎಡ್ಡೆ ಉಂಡು` (if human-transcribed) |
| `sentiment` | String | No | `positive`, `negative`, `neutral`, `unlabeled` |
| `dialect` | String | No | `Common Tulu` |
| `source` | String | No | `Field_Recording` |

### Supported Audio Formats

- Standard `.wav` (16-bit PCM, 16000 Hz mono recommended).
- `.mp3`, `.flac`, `.ogg` (auto-converted to 16kHz mono WAV during preprocessing).
- Minimum duration: `0.3` seconds.
- Files exceeding `30` seconds are automatically split into overlapping windows during audio preprocessing.

---

## 3. Loading Your Dataset in Python

```python
from src.unified_interface import load_text_dataset, preprocess_audio

# Load and validate your text dataset
df, records, report = load_text_dataset("data/raw/text/my_dataset.csv")

if report and not report.is_clean:
    print(f"Validation warnings found: {report.missing_fields_summary}")

# Process an audio file
res = preprocess_audio("data/raw/audio/my_recording.wav")
print(f"Standardized audio saved to: {res.processed_path}")
```

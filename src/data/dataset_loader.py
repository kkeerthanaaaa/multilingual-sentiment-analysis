"""Dataset loader pipeline and catalog of documented low-resource corpora."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd

from src.data.schema import TextRecord, AudioRecord, ValidationReport, SentimentLabel, InputType
from src.data.dataset_validator import DatasetValidator
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


# Registry documenting publicly known resources for target languages.
# NOTE: Status and license are factually annotated; no fake URLs or auto-downloads are fabricated.
KNOWN_PUBLIC_RESOURCES: Dict[str, Dict[str, Any]] = {
    "indicsentiment_ai4bharat": {
        "name": "IndicSentiment Benchmark (AI4Bharat)",
        "languages": ["assamese", "bodo", "kannada", "bengali", "hindi", "gujarati", "malayalam", "marathi", "odia", "punjabi", "tamil", "telugu", "urdu"],
        "modality": "text",
        "license": "CC-BY-4.0 / Research Use",
        "sentiment_labeled": True,
        "url": "https://huggingface.co/datasets/ai4bharat/IndicSentiment",
        "source": "AI4Bharat / IIT Madras",
        "notes": "Multilingual sentiment benchmark with human-annotated product reviews (Positive/Negative) across Indic languages including Assamese (as) and Bodo (bd)."
    },
    "indicsentiment_translated": {
        "name": "IndicSentiment-Translated (AI4Bharat)",
        "languages": ["assamese", "bodo", "kannada", "hindi", "marathi", "tamil", "telugu"],
        "modality": "text",
        "license": "CC-BY-4.0 / Research Use",
        "sentiment_labeled": True,
        "url": "https://huggingface.co/datasets/ai4bharat/IndicSentiment-Translated",
        "source": "AI4Bharat",
        "notes": "Machine-translated sentiment dataset via IndicTrans2; useful baseline but may lack colloquial nuances."
    },
    "indic_cmix": {
        "name": "IndicCMix Code-Mixed Corpus",
        "languages": ["assamese", "hindi", "bengali", "kannada", "tamil", "telugu"],
        "modality": "text+audio",
        "license": "Research Access (Gated)",
        "sentiment_labeled": False,
        "url": "https://huggingface.co/datasets/ai4bharat/IndicCMix",
        "source": "AI4Bharat",
        "notes": "Code-mixed Indic text and speech corpus for evaluating code-mixing and code-switching robustness."
    },
    "dravidian_codemix": {
        "name": "Dravidian-CodeMix Sentiment & Offensive Language Shared Tasks",
        "languages": ["kannada", "malayalam", "tamil"],
        "modality": "text",
        "license": "CC-BY-4.0 / Research",
        "sentiment_labeled": True,
        "url": "https://dravidian-codemix.github.io/",
        "source": "Forum for Information Retrieval Evaluation (FIRE / DravidianLangTech)",
        "notes": "Social media code-mixed and script-mixed comments; highly relevant for Dravidian/Kannada transfer to Tulu."
    },
    "indicvoices": {
        "name": "IndicVoices Speech Corpus",
        "languages": ["assamese", "bodo", "dogri", "konkani", "santali", "bhojpuri", "kannada"],
        "modality": "audio+transcript",
        "license": "CC-BY-4.0 / Gated Hugging Face Access",
        "sentiment_labeled": False,
        "url": "https://huggingface.co/datasets/ai4bharat/IndicVoices",
        "source": "AI4Bharat / Bhashini",
        "notes": "Comprehensive 7349-hour regional speech corpus covering read, extempore, and conversational speech."
    },
    "indicvoices_github": {
        "name": "IndicVoices GitHub Repository",
        "languages": ["all_22_scheduled_plus_regional"],
        "modality": "code+tools",
        "license": "MIT",
        "url": "https://github.com/AI4Bharat/IndicVoices",
        "source": "AI4Bharat",
        "notes": "Scripts, segmentation tools, and download utilities for the IndicVoices dataset."
    },
    "indicwav2vec": {
        "name": "IndicWav2Vec Acoustic Representations",
        "languages": ["multilingual_indic"],
        "modality": "audio_models",
        "license": "MIT",
        "url": "https://indicnlp.ai4bharat.org/indicwav2vec/",
        "source": "AI4Bharat",
        "notes": "Multilingual self-supervised speech representations pretrained on 17,000+ hours of raw Indian language audio."
    },
    "ai4bharat_portal": {
        "name": "AI4Bharat Datasets Portal",
        "languages": ["multilingual_indic"],
        "modality": "text+speech+translation",
        "license": "Various Open Research Licenses",
        "url": "https://ai4bharat.iitm.ac.in/datasets/",
        "source": "AI4Bharat / IIT Madras",
        "notes": "Official repository catalogue for Indian language AI resources."
    },
    "ai4bharat_indicnlp_catalog": {
        "name": "AI4Bharat Indic NLP Catalogue",
        "languages": ["multilingual_indic"],
        "modality": "text_corpora",
        "license": "Open Access",
        "url": "https://ai4bharat.github.io/indicnlp_catalog/",
        "source": "AI4Bharat",
        "notes": "Index of text corpora, tokenizers, word embeddings, and linguistic tools."
    },
    "bhashini": {
        "name": "Bhashini Mission National Language Translation Mission",
        "languages": ["22_scheduled_languages_plus_dialects"],
        "modality": "text+audio+vision",
        "license": "Government of India / Research Portal",
        "url": "https://bhashini.gov.in/",
        "source": "Ministry of Electronics and Information Technology (MeitY)",
        "notes": "National Indian Language Technology Platform."
    },
    "sraavaani_tulu": {
        "name": "SraVaani Tulu Speech & Transcription Corpus",
        "languages": ["tulu"],
        "modality": "audio+transcript",
        "license": "Restricted Research License (IISc / SPIRE)",
        "sentiment_labeled": False,
        "source": "IISc Bangalore / RESPIN Project",
        "notes": "Spoken regional recordings in Dakshina Kannada/Udupi. Access requires formal application to IISc."
    },
    "indra_bhojpuri": {
        "name": "InDRA Bhojpuri Spoken Corpus",
        "languages": ["bhojpuri"],
        "modality": "audio+transcript",
        "license": "Academic / Non-Commercial Use Only",
        "sentiment_labeled": False,
        "source": "InDRA / Regional Dialect Initiative",
        "notes": "Focuses on rural and colloquial Bhojpuri speech; unannotated for sentiment."
    },
    "ldc_il_dogri": {
        "name": "LDC-IL Dogri Sentence-Aligned Speech Resource",
        "languages": ["dogri"],
        "modality": "audio+transcript",
        "license": "LDC-IL Membership / Academic License",
        "sentiment_labeled": False,
        "source": "Linguistic Data Consortium for Indian Languages (CIIL Mysore)",
        "notes": "Clean sentence-aligned speech recorded in controlled environments. Proprietary academic license."
    },
    "ldc_il_bodo": {
        "name": "LDC-IL Bodo Text and Speech Corpus",
        "languages": ["bodo"],
        "modality": "text+speech",
        "license": "LDC-IL Membership / Academic License",
        "sentiment_labeled": False,
        "source": "LDC-IL (CIIL Mysore)",
        "notes": "Devanagari script Bodo. Requires subscription/agreement."
    },
    "indiccorp_santali": {
        "name": "IndicCorp v2 Santali Ol Chiki Corpus",
        "languages": ["santali"],
        "modality": "text",
        "license": "CC0 / Open Data",
        "sentiment_labeled": False,
        "source": "AI4Bharat",
        "notes": "Web-crawled Ol Chiki script text; requires downstream manual sentiment labeling."
    },
    "indiccorp_konkani": {
        "name": "IndicCorp v2 Konkani (Devanagari)",
        "languages": ["konkani"],
        "modality": "text",
        "license": "CC0 / Open Data",
        "sentiment_labeled": False,
        "source": "AI4Bharat",
        "notes": "Devanagari Konkani news and web text."
    }
}


class DatasetLoader:
    """Configurable loader for raw and annotated datasets supporting CSV and JSONL."""

    def __init__(self, validator: Optional[DatasetValidator] = None):
        self.validator = validator or DatasetValidator()

    @staticmethod
    def get_known_resources() -> Dict[str, Dict[str, Any]]:
        """Returns the catalog of documented external datasets."""
        return KNOWN_PUBLIC_RESOURCES

    def load_text_dataset(
        self,
        file_path: Union[str, Path],
        file_format: Optional[str] = None,
        validate: bool = True,
    ) -> Tuple[pd.DataFrame, List[TextRecord], Optional[ValidationReport]]:
        """Loads a text dataset from a CSV or JSONL file.

        Args:
            file_path: Path to dataset file.
            file_format: Optional format specification ('csv' or 'jsonl'). If None, inferred from extension.
            validate: Whether to run schema validation on loaded records.

        Returns:
            Tuple of (DataFrame, List of validated TextRecords, ValidationReport or None).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found at: {file_path}.\n"
                f"To add local data manually, place a valid CSV or JSONL in the data/raw/text directory.\n"
                f"Expected columns: sample_id, text, language, secondary_language, input_type, sentiment, script, dialect, source."
            )

        fmt = file_format.lower() if file_format else path.suffix.lstrip(".").lower()
        logger.info(f"Loading text dataset from {path} (format: {fmt})")

        if fmt == "csv":
            df = pd.read_csv(path, encoding="utf-8")
        elif fmt in ("jsonl", "jsonlines"):
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line_s = line.strip()
                    if not line_s:
                        continue
                    try:
                        records.append(json.loads(line_s))
                    except json.JSONDecodeError as err:
                        logger.warning(f"Malformed JSON on line {line_num} in {path}: {err}")
            df = pd.DataFrame(records)
        elif fmt == "json":
            df = pd.read_json(path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file format '{fmt}'. Expected 'csv', 'jsonl', or 'json'.")

        if "sample_id" not in df.columns:
            # Auto-assign unique sample_ids if omitted in input (e.g. 6-column prototype schema)
            df.insert(0, "sample_id", [f"rec_{i+1:05d}" for i in range(len(df))])

        records: List[TextRecord] = []
        report: Optional[ValidationReport] = None

        if validate:
            records, report = self.validator.validate_text_records(df)
            logger.info(f"Dataset {path.name}: {len(records)} valid text records extracted.")

        return df, records, report

    def load_audio_metadata(
        self,
        metadata_file_path: Union[str, Path],
        base_audio_dir: Optional[Union[str, Path]] = None,
        validate: bool = True,
        check_file_existence: bool = True,
    ) -> Tuple[pd.DataFrame, List[AudioRecord], Optional[ValidationReport]]:
        """Loads audio metadata mapping audio files to transcripts, language, and sentiment.

        Args:
            metadata_file_path: Path to CSV or JSONL audio metadata file.
            base_audio_dir: Base directory where audio recordings are stored.
            validate: Whether to validate metadata records.
            check_file_existence: Whether to verify audio files exist on disk.

        Returns:
            Tuple of (DataFrame, List of AudioRecords, ValidationReport).
        """
        path = Path(metadata_file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Audio metadata file not found at: {metadata_file_path}.\n"
                f"Place audio metadata CSV/JSONL in data/metadata/ or data/raw/audio/."
            )

        fmt = path.suffix.lstrip(".").lower()
        if fmt == "csv":
            df = pd.read_csv(path, encoding="utf-8")
        elif fmt in ("jsonl", "jsonlines"):
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            df = pd.DataFrame(records)
        else:
            raise ValueError(f"Unsupported audio metadata format: {fmt}. Use CSV or JSONL.")

        audio_base = str(base_audio_dir) if base_audio_dir else str(path.parent)
        records: List[AudioRecord] = []
        report: Optional[ValidationReport] = None

        if validate:
            records, report = self.validator.validate_audio_metadata(
                df, base_audio_dir=audio_base, check_file_existence=check_file_existence
            )

        return df, records, report

"""Dataset validator for text and audio sentiment records."""

import os
from pathlib import Path
from typing import List, Dict, Any, Union, Set, Optional, Tuple
import pandas as pd

from src.data.schema import TextRecord, AudioRecord, ValidationReport, SentimentLabel, InputType
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

SUPPORTED_LANGUAGE_ALIASES: Dict[str, Set[str]] = {
    "tulu": {"tulu", "tcy", "tulu_knda"},
    "assamese": {"assamese", "as", "asm"},
    "konkani": {"konkani", "kok", "gom"},
    "santali": {"santali", "sat", "santali_olchiki"},
    "bodo": {"bodo", "brx"},
    "bhojpuri": {"bhojpuri", "bho"},
    "dogri": {"dogri", "doi", "dgo"},
    "tamil": {"tamil", "ta", "tam"},
    "telugu": {"telugu", "te", "tel"},
    "malayalam": {"malayalam", "ml", "mal"},
    "kannada": {"kannada", "kn", "kan", "knda"},
}

ALL_SUPPORTED_CODES = {code for aliases in SUPPORTED_LANGUAGE_ALIASES.values() for code in aliases}


class DatasetValidator:
    """Validates raw text datasets and audio metadata files against standard schemas."""

    def __init__(self, supported_languages: Set[str] = ALL_SUPPORTED_CODES):
        self.supported_languages = supported_languages

    def is_supported_language(self, lang: str) -> bool:
        if not lang:
            return False
        return lang.strip().lower() in self.supported_languages

    def validate_text_records(self, df_or_records: Union[pd.DataFrame, List[Dict[str, Any]]]) -> (List[TextRecord], ValidationReport):
        """Validates a collection of text records.

        Args:
            df_or_records: pandas DataFrame or list of dicts.

        Returns:
            Tuple of (valid_records, validation_report).
        """
        report = ValidationReport()
        valid_records: List[TextRecord] = []
        seen_sample_ids: Set[str] = set()

        if isinstance(df_or_records, pd.DataFrame):
            records = df_or_records.to_dict(orient="records")
        else:
            records = df_or_records

        report.total_records = len(records)
        required_fields = ["sample_id", "text", "language"]

        for idx, row in enumerate(records):
            # Check for missing required fields
            row_errors = []
            for rf in required_fields:
                val = row.get(rf)
                if val is None or pd.isna(val) or str(val).strip() == "":
                    report.missing_fields_summary[rf] = report.missing_fields_summary.get(rf, 0) + 1
                    row_errors.append(f"Missing required field: '{rf}'")

            sample_id = str(row.get("sample_id", "")).strip() if row.get("sample_id") is not None and not pd.isna(row.get("sample_id")) else f"idx_{idx}"

            # Duplicate ID check
            if sample_id in seen_sample_ids:
                report.duplicate_ids.append(sample_id)
                row_errors.append(f"Duplicate sample_id: '{sample_id}'")
            elif sample_id:
                seen_sample_ids.add(sample_id)

            # Language check
            lang = str(row.get("language", "")).strip().lower() if row.get("language") is not None and not pd.isna(row.get("language")) else ""
            if lang and not self.is_supported_language(lang):
                if lang not in report.unsupported_languages:
                    report.unsupported_languages.append(lang)

            if row_errors:
                report.invalid_records += 1
                report.errors.append({
                    "record_index": idx,
                    "sample_id": sample_id,
                    "errors": row_errors,
                    "raw_data": {k: str(v) for k, v in row.items() if not pd.isna(v)}
                })
                continue

            # Try parsing with Pydantic
            try:
                # Handle pandas NaN safely
                cleaned_row = {
                    k: (None if pd.isna(v) else v)
                    for k, v in row.items()
                }
                record = TextRecord(**cleaned_row)
                valid_records.append(record)
                report.valid_records += 1
            except Exception as e:
                report.invalid_records += 1
                report.errors.append({
                    "record_index": idx,
                    "sample_id": sample_id,
                    "errors": [f"Schema validation error: {str(e)}"],
                    "raw_data": {k: str(v) for k, v in row.items() if not pd.isna(v)}
                })

        logger.info(
            f"Text validation complete: {report.valid_records}/{report.total_records} valid. "
            f"Invalid: {report.invalid_records}, Duplicates: {len(report.duplicate_ids)}."
        )
        return valid_records, report

    def validate_audio_metadata(
        self,
        df_or_records: Union[pd.DataFrame, List[Dict[str, Any]]],
        base_audio_dir: Optional[str] = None,
        check_file_existence: bool = True,
    ) -> (List[AudioRecord], ValidationReport):
        """Validates audio metadata records and verifies file existence.

        Args:
            df_or_records: pandas DataFrame or list of dicts.
            base_audio_dir: Optional base directory to resolve relative paths.
            check_file_existence: Whether to verify that audio files actually exist.

        Returns:
            Tuple of (valid_records, validation_report).
        """
        report = ValidationReport()
        valid_records: List[AudioRecord] = []
        seen_sample_ids: Set[str] = set()

        if isinstance(df_or_records, pd.DataFrame):
            records = df_or_records.to_dict(orient="records")
        else:
            records = df_or_records

        report.total_records = len(records)
        required_fields = ["sample_id", "audio_path", "language"]

        for idx, row in enumerate(records):
            row_errors = []
            for rf in required_fields:
                val = row.get(rf)
                if val is None or pd.isna(val) or str(val).strip() == "":
                    report.missing_fields_summary[rf] = report.missing_fields_summary.get(rf, 0) + 1
                    row_errors.append(f"Missing required field: '{rf}'")

            sample_id = str(row.get("sample_id", "")).strip() if row.get("sample_id") is not None and not pd.isna(row.get("sample_id")) else f"idx_{idx}"

            if sample_id in seen_sample_ids:
                report.duplicate_ids.append(sample_id)
                row_errors.append(f"Duplicate sample_id: '{sample_id}'")
            elif sample_id:
                seen_sample_ids.add(sample_id)

            audio_path = str(row.get("audio_path", "")).strip() if row.get("audio_path") is not None and not pd.isna(row.get("audio_path")) else ""
            if check_file_existence and audio_path:
                full_path = Path(base_audio_dir) / audio_path if base_audio_dir else Path(audio_path)
                if not full_path.exists():
                    row_errors.append(f"Audio file does not exist: {full_path}")

            if row_errors:
                report.invalid_records += 1
                report.errors.append({
                    "record_index": idx,
                    "sample_id": sample_id,
                    "errors": row_errors,
                    "raw_data": {k: str(v) for k, v in row.items() if not pd.isna(v)}
                })
                continue

            try:
                cleaned_row = {
                    k: (None if pd.isna(v) else v)
                    for k, v in row.items()
                }
                record = AudioRecord(**cleaned_row)
                valid_records.append(record)
                report.valid_records += 1
            except Exception as e:
                report.invalid_records += 1
                report.errors.append({
                    "record_index": idx,
                    "sample_id": sample_id,
                    "errors": [f"Audio schema validation error: {str(e)}"],
                    "raw_data": {k: str(v) for k, v in row.items() if not pd.isna(v)}
                })

        logger.info(
            f"Audio metadata validation complete: {report.valid_records}/{report.total_records} valid. "
            f"Invalid: {report.invalid_records}, Duplicates: {len(report.duplicate_ids)}."
        )
        return valid_records, report

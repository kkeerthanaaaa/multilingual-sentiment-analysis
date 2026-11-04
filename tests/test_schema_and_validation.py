"""Unit tests for schema definitions, record validation, and duplicate detection."""

import pytest
import pandas as pd
from src.data.schema import TextRecord, AudioRecord, SentimentLabel, InputType, ValidationReport
from src.data.dataset_validator import DatasetValidator


def test_text_record_valid_creation():
    """Tests creating a valid TextRecord with required and optional fields."""
    rec = TextRecord(
        sample_id="tcy_001",
        text="ದಾನೆ ಮಾರ್ರೆ ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು!",
        language="tulu",
        input_type=InputType.DIRECT,
        sentiment=SentimentLabel.POSITIVE,
        script="Kannada",
        dialect="Common Tulu",
    )
    assert rec.sample_id == "tcy_001"
    assert rec.language == "tulu"
    assert rec.sentiment == SentimentLabel.POSITIVE
    assert rec.input_type == InputType.DIRECT


def test_text_record_missing_required_fields():
    """Tests validation error when required fields are missing."""
    with pytest.raises(Exception):
        TextRecord(sample_id="", text="Sample text", language="tulu")

    with pytest.raises(Exception):
        TextRecord(sample_id="id_1", text="", language="tulu")

    with pytest.raises(Exception):
        TextRecord(sample_id="id_1", text="Sample", language="")


def test_sentiment_label_parsing():
    """Tests flexible sentiment label parsing."""
    assert SentimentLabel.from_str("positive") == SentimentLabel.POSITIVE
    assert SentimentLabel.from_str("pos") == SentimentLabel.POSITIVE
    assert SentimentLabel.from_str("NEGATIVE") == SentimentLabel.NEGATIVE
    assert SentimentLabel.from_str("neutral") == SentimentLabel.NEUTRAL
    assert SentimentLabel.from_str("") == SentimentLabel.UNLABELED
    assert SentimentLabel.from_str(None) == SentimentLabel.UNLABELED

    with pytest.raises(ValueError):
        SentimentLabel.from_str("invalid_label")


def test_validator_with_missing_and_invalid_records():
    """Tests validator handling missing fields, duplicates, and unsupported languages."""
    validator = DatasetValidator()
    raw_data = [
        {"sample_id": "rec_01", "text": "Valid text", "language": "bhojpuri", "sentiment": "positive"},
        {"sample_id": "rec_02", "text": "", "language": "tulu"},  # Missing text
        {"sample_id": "rec_01", "text": "Duplicate id text", "language": "dogri"},  # Duplicate sample_id
        {"sample_id": "rec_03", "text": "Valid", "language": "klingon"},  # Unsupported language
        {"sample_id": None, "text": "No ID", "language": "assamese"},  # Missing ID
    ]

    valid_recs, report = validator.validate_text_records(raw_data)

    assert report.total_records == 5
    assert len(valid_recs) == 2  # rec_01 and rec_03 (unsupported language still valid schema record but flagged)
    assert "rec_01" in report.duplicate_ids
    assert report.invalid_records >= 2
    assert "klingon" in report.unsupported_languages
    assert not report.is_clean


def test_audio_record_validation():
    """Tests audio record schema validation."""
    rec = AudioRecord(
        sample_id="aud_01",
        audio_path="data/raw/audio/synthetic_tcy_sample.wav",
        language="tulu",
        sentiment=SentimentLabel.POSITIVE,
    )
    assert rec.sample_id == "aud_01"
    assert rec.audio_path == "data/raw/audio/synthetic_tcy_sample.wav"

    with pytest.raises(Exception):
        AudioRecord(sample_id="aud_02", audio_path="", language="tulu")

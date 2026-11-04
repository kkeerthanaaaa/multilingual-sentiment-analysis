"""Unit tests for the 10 unified interface functions across Modules 1 & 2."""

import pytest
from pathlib import Path
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
from src.audio_processing.sraavaani_asr import TranscriptionStatus


def test_unified_load_text_dataset():
    """Tests load_text_dataset() loading CSV dataset."""
    df, records, report = load_text_dataset("data/raw/text/synthetic_sample_text_dataset.csv")
    assert not df.empty
    assert len(records) > 0
    assert report is not None
    assert report.total_records == len(df)


def test_unified_preprocess_text():
    """Tests preprocess_text() on regional text."""
    res = preprocess_text("ದಾನೆ ಮಾರ್ರೆ ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು! 😊")
    assert res.processed_text == "ದಾನೆ ಮಾರ್ರೆ ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು! 😊"
    assert "😊" in res.detected_emojis


def test_unified_identify_language():
    """Tests identify_language() function."""
    res = identify_language("ᱱᱚᱣᱟ ᱯᱷᱤᱞᱢ ᱟᱹᱰᱤ ᱵᱷᱟᱹᱜᱤ ᱜᱮᱭᱟ!")
    assert res.primary_language == "santali"
    assert res.primary_script == "Ol Chiki"


def test_unified_detect_code_mixing():
    """Tests detect_code_mixing() function."""
    is_mixed, ratio, sec_lang = detect_code_mixing("Movie was bad, ಎಂಕ್ ಇಷ್ಟ ಇಜ್ಜಿ at all!", code_mixing_threshold=0.15)
    assert is_mixed is True
    assert ratio > 0.15


def test_unified_validate_audio():
    """Tests validate_audio() function."""
    inspection = validate_audio("data/raw/audio/synthetic_tcy_sample.wav")
    assert inspection.is_valid is True
    assert inspection.sample_rate == 16000


def test_unified_preprocess_audio():
    """Tests preprocess_audio() function."""
    res = preprocess_audio(
        "data/raw/audio/synthetic_tcy_sample.wav",
        output_dir="data/processed/unified_audio_test",
    )
    assert res.error is None
    assert Path(res.processed_path).exists()


def test_unified_transcribe_audio():
    """Tests transcribe_audio() returning explicit UNAVAILABLE status without fabrication."""
    res = transcribe_audio("data/raw/audio/synthetic_tcy_sample.wav", language="tulu")
    assert res.status == TranscriptionStatus.UNAVAILABLE
    assert res.transcript is None


def test_unified_extract_acoustic_features():
    """Tests extract_acoustic_features() returning structured acoustic data."""
    output = extract_acoustic_features(["data/raw/audio/synthetic_tcy_sample.wav"])
    assert output.num_samples == 1
    assert not output.features_df.empty

"""Unit tests for audio inspection, format validation, standardization, and segmentation."""

import os
from pathlib import Path
import pytest
from src.preprocessing.audio_preprocessing import AudioPreprocessor


def test_audio_inspection_valid_file():
    """Tests inspecting a valid 16kHz WAV recording."""
    preprocessor = AudioPreprocessor()
    inspection = preprocessor.inspect_audio("data/raw/audio/synthetic_tcy_sample.wav")

    assert inspection.is_valid is True
    assert inspection.sample_rate == 16000
    assert inspection.channels == 1
    assert 1.4 <= inspection.duration_seconds <= 1.6
    assert inspection.error_message is None


def test_empty_audio_file_rejection():
    """Tests that a 0-byte file is identified as invalid."""
    preprocessor = AudioPreprocessor()
    inspection = preprocessor.inspect_audio("data/raw/audio/empty_invalid_sample.wav")

    assert inspection.is_valid is False
    assert "empty" in inspection.error_message.lower()


def test_nonexistent_audio_file():
    """Tests that non-existent paths return an informative error."""
    preprocessor = AudioPreprocessor()
    inspection = preprocessor.inspect_audio("data/raw/audio/non_existent_file.wav")

    assert inspection.is_valid is False
    assert "does not exist" in inspection.error_message.lower()


def test_unsupported_audio_extension():
    """Tests that unsupported file extensions are rejected."""
    preprocessor = AudioPreprocessor()
    inspection = preprocessor.inspect_audio("data/raw/audio/test_audio.txt")

    assert inspection.is_valid is False
    assert "unsupported" in inspection.error_message.lower()


def test_audio_preprocessing_and_standardization():
    """Tests conversion and standardization pipeline."""
    preprocessor = AudioPreprocessor(target_sample_rate=16000)
    out_dir = "data/processed/test_audio"

    result = preprocessor.preprocess_audio_file(
        audio_path="data/raw/audio/synthetic_tcy_sample.wav",
        output_dir=out_dir,
        output_filename="test_proc_tcy.wav",
        apply_preemphasis=True,
    )

    assert result.error is None
    assert Path(result.processed_path).exists()
    assert result.target_sample_rate == 16000
    assert result.applied_preemphasis is True
    assert result.was_normalized is True


def test_long_audio_segmentation():
    """Tests automatic windowed segmentation for audio exceeding max_duration."""
    # synthetic_long_sample.wav is 35s long; set max_duration=30s and segment_duration=15s
    preprocessor = AudioPreprocessor(
        max_duration_seconds=30.0,
        segment_duration_seconds=15.0,
        segment_overlap_seconds=1.0,
    )
    out_dir = "data/processed/test_audio"

    result = preprocessor.preprocess_audio_file(
        audio_path="data/raw/audio/synthetic_long_sample.wav",
        output_dir=out_dir,
        segment_if_long=True,
    )

    assert result.error is None
    assert len(result.segments) >= 2
    for seg in result.segments:
        assert Path(seg.output_path).exists()
        assert seg.duration_seconds <= 15.1

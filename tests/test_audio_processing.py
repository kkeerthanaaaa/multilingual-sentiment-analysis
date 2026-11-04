"""Unit tests for audio transcription (SraVaani) and feature extraction (XLS-R & openSMILE)."""

import os
from pathlib import Path
import pytest
from src.audio_processing.sraavaani_asr import SraVaaniASRClient, TranscriptionStatus
from src.audio_processing.opensmile_features import OpenSMILEFeatureExtractor
from src.audio_processing.xlsr_features import XLSRFeatureExtractor


def test_sraavaani_unconfigured_service_returns_unavailable():
    """Verifies that an unconfigured SraVaani client returns UNAVAILABLE without fake transcription."""
    client = SraVaaniASRClient(api_endpoint=None, api_key=None)
    result = client.transcribe(
        audio_path="data/raw/audio/synthetic_tcy_sample.wav",
        language="tulu",
    )

    assert result.status == TranscriptionStatus.UNAVAILABLE
    assert result.transcript is None  # Integrity: MUST NOT fabricate fake transcription
    assert "service is currently unconfigured" in result.error
    assert "SRAVAANI_API_ENDPOINT" in result.error


def test_sraavaani_unsupported_language_rejection():
    """Verifies that unsupported languages are rejected with clear status."""
    client = SraVaaniASRClient(api_endpoint="http://example.com/api", api_key="test_key")
    result = client.transcribe(
        audio_path="data/raw/audio/synthetic_tcy_sample.wav",
        language="french",
    )

    assert result.status == TranscriptionStatus.UNSUPPORTED_LANGUAGE
    assert result.transcript is None
    assert "not supported" in result.error


def test_sraavaani_invalid_audio_rejection():
    """Verifies that empty or invalid audio is rejected before sending external requests."""
    client = SraVaaniASRClient(api_endpoint="http://example.com/api", api_key="test_key")
    result = client.transcribe(
        audio_path="data/raw/audio/empty_invalid_sample.wav",
        language="tulu",
    )

    assert result.status == TranscriptionStatus.INVALID_AUDIO
    assert result.transcript is None


def test_acoustic_feature_extraction_on_synthetic_audio():
    """Tests extracting acoustic features (F0, energy, spectral) on valid audio."""
    extractor = OpenSMILEFeatureExtractor(feature_set="eGeMAPSv02")
    output = extractor.extract_batch([
        "data/raw/audio/synthetic_tcy_sample.wav",
        "data/raw/audio/synthetic_bho_sample.wav",
    ])

    assert output.num_samples == 2
    assert not output.features_df.empty
    assert len(output.feature_names) > 0
    assert "sample_id" in output.features_df.columns

    # Save features to CSV and verify file output
    out_csv = "data/processed/test_acoustic_features.csv"
    saved_path = extractor.save_features(output, out_csv)
    assert Path(saved_path).exists()
    assert os.path.getsize(saved_path) > 50


def test_xlsr_extractor_handles_short_audio():
    """Verifies that XLS-R feature extractor handles excessively short clips gracefully."""
    extractor = XLSRFeatureExtractor(min_duration_seconds=5.0)  # artificially high min duration
    emb, err = extractor.extract_single_audio("data/raw/audio/synthetic_tcy_sample.wav")

    assert emb is None
    assert "below minimum" in err

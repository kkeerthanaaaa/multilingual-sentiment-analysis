"""Tests for real-time dataset ingestion, prototype schema validation, and model training."""

import os
from pathlib import Path
import pytest
import pandas as pd

from src.data.schema import TextRecord, SentimentLabel, InputType, LanguageStyle
from src.data.dataset_loader import DatasetLoader
from src.text_processing.hybrid_classifier import HybridSentimentClassifier
from src.text_processing.indicbert import IndicBERTProcessor
from src.training.train_sentiment import train_sentiment_model


def test_prototype_schema_compatibility():
    """Tests that the prototype 6-column schema validates and parses correctly."""
    data = [
        {
            "language": "tulu",
            "text": "ದಾನೆ ಮಾರ್ರೆ ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು!",
            "input_type": "Text",
            "language_style": "Slang-dialect-heavy",
            "audio_path": "",
            "sentiment": "Positive",
        },
        {
            "language": "bhojpuri",
            "text": "का बात बा भाई!",
            "input_type": "Audio",
            "language_style": "Slang-dialect-heavy",
            "audio_path": "data/raw/audio/synthetic_bho_sample.wav",
            "sentiment": "Positive",
        },
        {
            "language": "assamese",
            "text": "এইটো বহুত ভাল!",
            "input_type": "Text",
            "language_style": "Direct",
            "audio_path": "",
            "sentiment": "Positive",
        },
    ]
    df = pd.DataFrame(data)

    loader = DatasetLoader()
    # Save temporary CSV
    tmp_path = Path("data/raw/text/test_proto_schema.csv")
    df.to_csv(tmp_path, index=False)

    try:
        loaded_df, recs, report = loader.load_text_dataset(tmp_path)
        assert len(recs) == 3
        assert report.is_clean
        assert recs[0].language == "tulu"
        assert recs[0].language_style == LanguageStyle.SLANG_DIALECT_HEAVY
        assert recs[0].sentiment == SentimentLabel.POSITIVE
        assert recs[1].input_type == InputType.AUDIO
        assert recs[1].audio_path == "data/raw/audio/synthetic_bho_sample.wav"
        assert recs[0].sample_id.startswith("rec_")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_real_time_multilingual_dataset_exists_and_conforms():
    """Verifies that the generated real-time dataset exists and has target language representation."""
    dataset_file = Path("data/raw/text/real_time_multilingual_sentiment_dataset.csv")
    assert dataset_file.exists(), "real_time_multilingual_sentiment_dataset.csv does not exist"

    df = pd.read_csv(dataset_file)
    expected_cols = {"language", "text", "input_type", "language_style", "audio_path", "sentiment"}
    assert expected_cols.issubset(set(df.columns))

    langs = set(df["language"].str.lower())
    assert "assamese" in langs
    assert "bodo" in langs
    assert "tulu" in langs
    assert "bhojpuri" in langs
    assert len(df) >= 50


def test_hybrid_sentiment_classifier_training_and_inference():
    """Tests fitting HybridSentimentClassifier and predicting on multilingual sentences."""
    train_texts = [
        "ದಾನೆ ಮಾರ್ರೆ ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು!",
        "ಈ ಸಿನಿಮಾ एकदम निक बा भाई, मजा आ गइल!",
        "বহুত ভাল লাগিল, খুব সুন্দৰ!",
        "ಎಂಕ್ ಈ ಮೂವಿ ಒಂತೆಲಾ ಇಷ್ಟ ಆಯಿಜಿ, ಬಾರಿ ಬೇಜಾರ್.",
        "एकदम बकवास पिक्चर, पईसा बरबाद!",
        "বেয়া অভিজ্ঞতা, একেবাৰে ভাল নহয়.",
    ]
    train_labels = ["positive", "positive", "positive", "negative", "negative", "negative"]

    clf = HybridSentimentClassifier()
    metrics = clf.fit(train_texts, train_labels)

    assert clf.is_trained
    assert metrics.overall_accuracy >= 0.80

    # Test inference
    preds = clf.predict(["ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು", "एकदम बकवास"])
    assert len(preds) == 2
    assert preds[0].predicted_label == "positive"
    assert preds[1].predicted_label == "negative"
    assert 0.0 <= preds[0].confidence <= 1.0


def test_indicbert_fine_tune_guard():
    """Tests that IndicBERT fine-tune guards against insufficient labeled data."""
    proc = IndicBERTProcessor()
    train_recs = [
        TextRecord(sample_id="1", text="text 1", language="assamese", sentiment=SentimentLabel.POSITIVE),
        TextRecord(sample_id="2", text="text 2", language="assamese", sentiment=SentimentLabel.NEGATIVE),
    ]
    res = proc.fine_tune(train_recs)
    assert not res["trained"]
    assert "Insufficient" in res["reason"]


def test_saved_model_checkpoint_loadable():
    """Tests that trained model checkpoint can be reloaded and used for inference."""
    model_file = Path("models/hybrid_sentiment/hybrid_sentiment_pipeline.joblib")
    assert model_file.exists(), "Model checkpoint does not exist"

    clf = HybridSentimentClassifier()
    clf.load("models/hybrid_sentiment")
    assert clf.is_trained

    preds = clf.predict(["দಾನೆ ಮಾರ್ರೆ ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು!"])
    assert len(preds) == 1
    assert preds[0].model_trained
    assert preds[0].predicted_label in ("positive", "negative", "neutral")

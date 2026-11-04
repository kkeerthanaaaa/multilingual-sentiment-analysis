"""Unit tests for bilingual and code-mixed input handling."""

import pytest
from src.preprocessing.language_detection import LanguageDetector
from src.unified_interface import detect_code_mixing


def test_code_mixed_latin_indic_sentence():
    """Tests sentences mixing Latin script (English) with Kannada script (Tulu)."""
    text = "Movie was very bad, ಎಂಕ್ ಇಷ್ಟ ಇಜ್ಜಿ at all!"
    is_mixed, ratio, sec_lang = detect_code_mixing(text, code_mixing_threshold=0.15)

    assert is_mixed is True
    assert ratio > 0.15
    assert sec_lang == "en"


def test_code_mixed_bhojpuri_english():
    """Tests code-mixing between English and Bhojpuri."""
    detector = LanguageDetector(code_mixing_threshold=0.15)
    text = "Today feeling super awesome, का बात बा bhai!"
    res = detector.detect(text)

    assert res.is_code_mixed is True
    assert "Latin" in res.detected_scripts
    assert "Devanagari" in res.detected_scripts
    assert res.secondary_language == "en"


def test_monolingual_sentence_not_flagged_as_mixed():
    """Pure monolingual sentence should not be flagged as code-mixed."""
    text = "এই চলচ্চিত্ৰখন সঁচাকৈয়ে বৰ ভাল লাগিল।"
    is_mixed, ratio, sec_lang = detect_code_mixing(text)

    assert is_mixed is False
    assert ratio < 0.20

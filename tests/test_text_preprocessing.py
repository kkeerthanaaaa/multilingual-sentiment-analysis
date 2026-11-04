"""Unit tests for text preprocessing, negation preservation, emojis, and deduplication."""

import pytest
from src.preprocessing.text_preprocessing import TextPreprocessor


def test_unicode_normalization_preserves_scripts():
    """Verifies Unicode NFC normalization preserves Indic characters without corruption."""
    preprocessor = TextPreprocessor()

    # Assamese with distinctive character ৰ
    asm_text = "মই চলচ্চিত্ৰখন চালোঁ, বৰ ভাল লাগিল।"
    res_asm = preprocessor.preprocess(asm_text)
    assert "ৰ" in res_asm.processed_text
    assert res_asm.original_text == asm_text

    # Ol Chiki (Santali)
    sat_text = "ᱱᱚᱣᱟ ᱯᱷᱤᱞᱢ ᱟᱹᱰᱤ ᱵᱷᱟᱹᱜᱤ ᱜᱮᱭᱟ!"
    res_sat = preprocessor.preprocess(sat_text)
    assert "ᱵᱷᱟᱹᱜᱤ" in res_sat.processed_text
    assert res_sat.processed_text == sat_text

    # Kannada script (Tulu)
    tcy_text = "ದಾನೆ ಮಾರ್ರೆ ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು"
    res_tcy = preprocessor.preprocess(tcy_text)
    assert "ದಾನೆ" in res_tcy.processed_text


def test_preservation_of_negations_across_languages():
    """Verifies that regional negation terms are explicitly identified and preserved."""
    preprocessor = TextPreprocessor()

    # Tulu negation 'ಇಜ್ಜಿ'
    res_tcy = preprocessor.preprocess("ಎಂಕ್ ಈ ಪೋಲಿಸ್ ಇಜ್ಜಿ ಇಷ್ಟ.")
    assert "ಇಜ್ಜಿ" in res_tcy.detected_negations
    assert "ಇಜ್ಜಿ" in res_tcy.processed_text

    # Assamese negation 'নহয়'
    res_asm = preprocessor.preprocess("এই কামটো নহয় একেবাৰে ভাল।")
    assert "নহয়" in res_asm.detected_negations

    # Bhojpuri negation 'नाहीं'
    res_bho = preprocessor.preprocess("हम नाहीं जाइब ओकरा साथे।")
    assert "नाहीं" in res_bho.detected_negations

    # Dogri negation 'नेईं'
    res_doi = preprocessor.preprocess("एह काम नेईं करणा भाई।")
    assert "नेईं" in res_doi.detected_negations

    # Santali negation 'ᱵᱟᱝ'
    res_sat = preprocessor.preprocess("ᱱᱚᱝᱠᱟ ᱫᱚ ᱵᱟᱝ ᱴᱷᱤᱠ ᱠᱟᱱᱟ।")
    assert "ᱵᱟᱝ" in res_sat.detected_negations


def test_preservation_of_emojis_and_sentiment_punctuation():
    """Verifies emojis and expressive punctuation are retained."""
    preprocessor = TextPreprocessor()

    text_with_emojis = "Super movie! 😊❤️🔥 But bad ending...!!!!"
    res = preprocessor.preprocess(text_with_emojis)

    assert "😊" in res.detected_emojis
    assert "❤️" in res.detected_emojis
    assert "🔥" in res.detected_emojis
    assert "!" in res.processed_text
    # Excessive runs of '!!!!' are normalized to '!' while preserving sentiment
    assert "!!!!" not in res.processed_text


def test_duplicate_and_near_duplicate_detection():
    """Tests exact hash deduplication and character n-gram near-duplicate detection."""
    preprocessor = TextPreprocessor(near_duplicate_threshold=0.85)
    preprocessor.reset_deduplication_cache()

    sample_1 = "ದಾನೆ ಮಾರ್ರೆ ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು!"
    res1 = preprocessor.preprocess(sample_1, sample_id="s1")
    assert not res1.is_duplicate

    # Exact duplicate (with extra spacing)
    sample_2 = "  ದಾನೆ ಮಾರ್ರೆ   ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು!  "
    res2 = preprocessor.preprocess(sample_2, sample_id="s2")
    assert res2.is_duplicate
    assert res2.duplicate_of_id == "s1"

    # Near duplicate with minor typo
    sample_3 = "ದಾನೆ ಮಾರ್ರೆ ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು"
    res3 = preprocessor.preprocess(sample_3, sample_id="s3")
    assert res3.is_duplicate


def test_separate_original_and_processed_storage():
    """Verifies original text and processed text are stored separately without destructive overwriting."""
    preprocessor = TextPreprocessor()
    messy_text = "   \t\n  का बात बा भाई!   😊  "
    res = preprocessor.preprocess(messy_text)

    assert res.original_text == messy_text
    assert res.processed_text == "का बात बा भाई! 😊"
    assert res.original_text != res.processed_text

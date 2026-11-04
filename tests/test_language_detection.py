"""Unit tests for language identification and ambiguity detection."""

import pytest
from src.preprocessing.language_detection import LanguageDetector


def test_ol_chiki_santali_deterministic_identification():
    """Santali in Ol Chiki script should be deterministically identified."""
    detector = LanguageDetector()
    text = "ᱱᱚᱣᱟ ᱯᱷᱤᱞᱢ ᱟᱹᱰᱤ ᱵᱷᱟᱹᱜᱤ ᱜᱮᱭᱟ!"
    res = detector.detect(text)

    assert res.primary_language == "santali"
    assert res.primary_script == "Ol Chiki"
    assert res.confidence >= 0.90
    assert not res.is_ambiguous


def test_assamese_distinctive_characters():
    """Assamese text containing distinctive characters ৰ or ৱ should be identified."""
    detector = LanguageDetector()
    text = "এই চলচ্চিত্ৰখন সঁচাকৈয়ে বৰ ভাল লাগিল।"
    res = detector.detect(text)

    assert res.primary_language == "assamese"
    assert res.primary_script == "Bengali"
    assert res.confidence >= 0.85
    assert not res.is_ambiguous


def test_tulu_kannada_script_disambiguation():
    """Kannada script with Tulu vocabulary markers should resolve to Tulu."""
    detector = LanguageDetector()
    text = "ದಾನೆ ಮಾರ್ರೆ ಮಸ್ತ್ ಎಡ್ಡೆ ಉಂಡು, ಬೋಡು ಇಜ್ಜಿ!"
    res = detector.detect(text)

    assert res.primary_language == "tulu"
    assert res.primary_script == "Kannada"
    assert res.confidence >= 0.80


def test_language_uncertainty_and_ambiguity_flagging():
    """Ambiguous or generic Devanagari text without clear markers must flag uncertainty."""
    detector = LanguageDetector(confidence_threshold=0.65)
    # Generic Devanagari sentence without distinct regional dialect markers
    text = "यह एक सामान्य वाक्य है जिसमें कोई स्पष्ट बोली नहीं है।"
    res = detector.detect(text)

    # Must transparently flag ambiguity rather than forcing an inaccurate regional label
    assert res.is_ambiguous is True
    assert res.review_recommended is True
    assert res.confidence < 0.65


def test_bhojpuri_lexical_markers():
    """Bhojpuri text with colloquial markers 'का बात बा' and 'नीक'."""
    detector = LanguageDetector()
    text = "का बात बा! ई गाना त एकदम नीक लागल।"
    res = detector.detect(text)

    assert res.primary_language == "bhojpuri"
    assert res.primary_script == "Devanagari"
    assert res.confidence >= 0.70

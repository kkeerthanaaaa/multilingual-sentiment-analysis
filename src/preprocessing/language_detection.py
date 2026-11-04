"""Hybrid language identification and code-mixing detection for low-resource Indian languages."""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
import langdetect
from langdetect import DetectorFactory

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Enforce deterministic results from langdetect's randomized seed
DetectorFactory.seed = 42

# Script Unicode Blocks
SCRIPT_RANGES = {
    "Ol Chiki": (0x1C50, 0x1C7F),     # Santali
    "Tamil": (0x0B80, 0x0BFF),        # Tamil
    "Telugu": (0x0C00, 0x0C7F),       # Telugu
    "Kannada": (0x0C80, 0x0CFF),      # Kannada, Tulu, Konkani
    "Malayalam": (0x0D00, 0x0D7F),    # Malayalam
    "Bengali": (0x0980, 0x09FF),      # Assamese, Bengali
    "Devanagari": (0x0900, 0x097F),   # Bhojpuri, Dogri, Bodo, Konkani, Hindi, Marathi
    "Latin": (0x0041, 0x007A),        # English, Romi Konkani, Transliterated
}

# Distinctive lexical markers for target regional and low-resource languages
LEXICAL_MARKERS: Dict[str, Set[str]] = {
    "tulu": {
        "ದಾನೆ", "ಬೋಡು", "ಇಜ್ಜಿ", "ಇಜ್ಜ", "ಆಯೆ", "ಯಾನ್", "ಎಂಕ್", "ಉಂಡು", "ಅತ್ತ", "ಎಡ್ಡೆ",
        "ಕುಡ್ಲ", "ಮಾರ್ರೆ", "ಲಾಯ್ಕ್", "ಪೊಂಡೆ", "ಬಾವ", "daane", "bodu", "ijji", "edde", "kudla"
    },
    "kannada": {
        "ಚೆನ್ನಾಗಿದೆ", "ತುಂಬಾ", "ಇಲ್ಲ", "ಖುಷಿ", "ಕೋಪ", "ಭಯ", "ಧನ್ಯವಾದ", "ಕೆಟ್ಟ", "ಸರಿ",
        "ಹೇಗಿದ್ದೀರಾ", "ಮಗಾ", "ಗುರು", "ಬೆಂಕಿ", "ಚಿಂದಿ", "chennagide", "tumba", "beku", "guru"
    },
    "tamil": {
        "வணக்கம்", "ரொம்ப", "நல்ல", "இல்லை", "கெட்ட", "மகிழ்ச்சி", "கோபம்", "பயம்",
        "நன்றி", "மச்சி", "தல", "மாப்ள", "கெத்து", "வெறித்தனம்", "romba", "nalla", "illai", "machi"
    },
    "telugu": {
        "బాగుంది", "లేదు", "చాలా", "కోపం", "భయం", "సంతోషం", "ధన్యవాదాలు", "చెడ్డ",
        "అవును", "మావా", "రచ్చ", "కిర్రాక్", "bagundi", "chala", "ledu", "mawa", "kirrak"
    },
    "malayalam": {
        "നല്ല", "വളരെ", "ഇല്ല", "സന്തോഷം", "ദേഷ്യം", "ഭയം", "നന്ദി", "മോശം",
        "ശരി", "സുഖമാണോ", "അളിയാ", "പൊളി", "തകർത്തു", "സീൻ", "മച്ചാനെ", "nalla", "valare", "illa", "aliya"
    },
    "assamese": {
        "ৰ", "ৱ", "নহয়", "নাই", "নালাগে", "নকৰোঁ", "নোৱাৰো", "ভাল", "বেয়া", "ধুনীয়া",
        "হেৰৌ", "লেঠা", "দেউতা", "nohoy", "dhuniya"
    },
    "konkani": {
        "हांव", "आसा", "ना", "व्हय", "बोरें", "वायट", "कसो", "कितें", "नाका", "नुसत",
        "borem", "sobit", "kitem", "nhay"
    },
    "santali": {
        "ᱵᱟᱝ", "ᱵᱟ", "ᱵᱟᱭ", "ᱟᱞᱚ", "ᱵᱷᱟᱹᱜᱤ", "ᱱᱟᱯᱟᱭ", "ᱨᱟᱹᱥᱠᱟᱹ", "ᱵᱟᱹᱲᱤᱡ", "ᱫᱩᱠᱷ", "ᱡᱚᱦᱟᱨ"
    },
    "bodo": {
        "दा", "गोजां", "मोजां", "गाज्रि", "नों", "आं", "दं", "नङಾ", "गैया", "हाया", "सबिन",
        "mojang", "gajri"
    },
    "bhojpuri": {
        "बा", "बानी", "रहल", "बाटे", "हमरा", "तोहरा", "काहे", "नाहीं", "नीक", "बेतूका",
        "रंगदार", "रंगबाज", "का बात बा", "गजबे", "neek", "rahal", "baat ba"
    },
    "dogri": {
        "एह", "ओह", "गे", "मिकी", "तुगी", "शा", "चंगा", "मंदा", "नेईं", "बधिया", "शैल",
        "कदा", "miki", "tugi", "changa", "nein"
    }
}


@dataclass
class LanguageIDResult:
    """Detailed result of language identification and code-mixing analysis."""
    text: str
    primary_language: str
    confidence: float
    secondary_language: Optional[str] = None
    is_code_mixed: bool = False
    code_mixing_ratio: float = 0.0
    detected_scripts: List[str] = field(default_factory=list)
    primary_script: str = "Unknown"
    is_ambiguous: bool = False
    review_recommended: bool = False
    detection_method: str = "unknown"
    notes: str = ""


class LanguageDetector:
    """Language identification engine designed for low-resource Indian languages.
    
    Combines Unicode script boundaries, regional lexical markers, and statistical n-gram models.
    Transparently flags uncertainty rather than forcing incorrect standard-language labels.
    """

    def __init__(self, confidence_threshold: float = 0.65, code_mixing_threshold: float = 0.20):
        self.confidence_threshold = confidence_threshold
        self.code_mixing_threshold = code_mixing_threshold

    @staticmethod
    def identify_script(char: str) -> Optional[str]:
        """Maps an individual character code to its Unicode script name."""
        cp = ord(char)
        for script_name, (start, end) in SCRIPT_RANGES.items():
            if start <= cp <= end:
                return script_name
        return None

    def analyze_scripts(self, text: str) -> Dict[str, float]:
        """Computes the distribution percentage of scripts in the text."""
        script_counts: Dict[str, int] = {}
        total_letters = 0

        for ch in text:
            script = self.identify_script(ch)
            if script:
                script_counts[script] = script_counts.get(script, 0) + 1
                total_letters += 1

        if total_letters == 0:
            return {}

        return {s: count / total_letters for s, count in script_counts.items()}

    def _check_lexical_markers(self, text: str) -> Dict[str, int]:
        """Counts distinctive lexical markers for each regional language."""
        text_lower = text.lower()
        tokens = set(re.findall(r"\w+", text_lower, flags=re.UNICODE))
        matches: Dict[str, int] = {}

        for lang, markers in LEXICAL_MARKERS.items():
            count = 0
            for marker in markers:
                if marker in tokens or (len(marker) > 2 and marker in text_lower):
                    count += 1
            if count > 0:
                matches[lang] = count

        return matches

    def detect(self, text: str) -> LanguageIDResult:
        """Identifies language, script, code-mixing, and ambiguity.

        Args:
            text: Input sentence or document.

        Returns:
            LanguageIDResult: Structured language identification result.
        """
        raw_text = str(text or "").strip()
        if not raw_text:
            return LanguageIDResult(
                text=raw_text,
                primary_language="unknown",
                confidence=0.0,
                is_ambiguous=True,
                review_recommended=True,
                notes="Empty text provided."
            )

        # Step 1: Analyze script composition
        script_dist = self.analyze_scripts(raw_text)
        sorted_scripts = sorted(script_dist.items(), key=lambda x: x[1], reverse=True)
        primary_script = sorted_scripts[0][0] if sorted_scripts else "Unknown"
        detected_scripts = [s for s, _ in sorted_scripts]

        # Step 2: Check for code-mixing
        is_code_mixed = False
        secondary_lang = None
        code_mix_ratio = 0.0

        if len(sorted_scripts) >= 2:
            second_script, ratio = sorted_scripts[1]
            if ratio >= self.code_mixing_threshold:
                is_code_mixed = True
                code_mix_ratio = ratio
                if second_script == "Latin":
                    secondary_lang = "en"  # Standard assumption for Latin code-mixing in Indian contexts

        # Step 3: Check regional lexical markers
        lexical_matches = self._check_lexical_markers(raw_text)

        # Step 4: Disambiguate based on script and markers
        primary_lang = "unknown"
        confidence = 0.5
        is_ambiguous = False
        review_recommended = False
        method = "heuristic"
        notes = ""

        # Case A: Ol Chiki script -> Deterministically Santali
        if primary_script == "Ol Chiki":
            primary_lang = "santali"
            confidence = 0.95
            method = "unicode_script_exact"
            notes = "Deterministically identified via Ol Chiki script block."

        # Case B: Tamil script -> Deterministically Tamil
        elif primary_script == "Tamil":
            primary_lang = "tamil"
            confidence = 0.95
            method = "unicode_script_exact"
            notes = "Deterministically identified via Tamil script block."

        # Case C: Telugu script -> Deterministically Telugu
        elif primary_script == "Telugu":
            primary_lang = "telugu"
            confidence = 0.95
            method = "unicode_script_exact"
            notes = "Deterministically identified via Telugu script block."

        # Case D: Malayalam script -> Deterministically Malayalam
        elif primary_script == "Malayalam":
            primary_lang = "malayalam"
            confidence = 0.95
            method = "unicode_script_exact"
            notes = "Deterministically identified via Malayalam script block."

        # Case E: Bengali script -> Assamese vs Bengali
        elif primary_script == "Bengali":
            # Distinctive Assamese letters U+09F0 (ৰ) and U+09F1 (ৱ)
            if "ৰ" in raw_text or "ৱ" in raw_text or lexical_matches.get("assamese", 0) > 0:
                primary_lang = "assamese"
                confidence = 0.90
                method = "script_distinctive_char"
                notes = "Identified as Assamese via distinctive characters (ৰ/ৱ) or lexical markers."
            else:
                # Could be Bengali or Assamese without distinctive chars
                primary_lang = "assamese"  # In our regional focus context
                confidence = 0.55
                is_ambiguous = True
                review_recommended = True
                method = "shared_script_fallback"
                notes = "Bengali/Assamese shared script. No distinct Assamese characters (ৰ/ৱ) found. Ambiguous."

        # Case F: Kannada script -> Tulu vs Kannada
        elif primary_script == "Kannada":
            if lexical_matches.get("tulu", 0) > 0:
                primary_lang = "tulu"
                confidence = 0.85
                method = "script_and_lexicon"
                notes = f"Identified as Tulu via Kannada script and {lexical_matches['tulu']} distinctive marker(s)."
            elif lexical_matches.get("kannada", 0) > 0:
                primary_lang = "kannada"
                confidence = 0.92
                method = "script_and_lexicon"
                notes = f"Identified as Kannada via Kannada script and {lexical_matches['kannada']} distinctive marker(s)."
            else:
                primary_lang = "kannada"
                confidence = 0.85
                method = "kannada_script_default"
                notes = "Identified as Kannada via native Kannada script block."

        # Case G: Devanagari script -> Bhojpuri, Dogri, Bodo, Konkani, Hindi
        elif primary_script == "Devanagari":
            # Evaluate regional lexical matches
            devanagari_langs = ["bhojpuri", "dogri", "bodo", "konkani"]
            matched_dev = {k: v for k, v in lexical_matches.items() if k in devanagari_langs}

            if matched_dev:
                top_match, count = max(matched_dev.items(), key=lambda x: x[1])
                primary_lang = top_match
                confidence = 0.82
                method = "devanagari_lexical_markers"
                notes = f"Identified as {top_match} via Devanagari script and {count} regional lexical marker(s)."
            else:
                # Try statistical detector fallback
                try:
                    stat_lang = langdetect.detect(raw_text)
                except Exception:
                    stat_lang = "hi"

                # langdetect only reports 'hi' or 'mr' for Devanagari
                primary_lang = "bhojpuri" if "बा" in raw_text else "devanagari_unclassified"
                confidence = 0.45
                is_ambiguous = True
                review_recommended = True
                method = "devanagari_ambiguity"
                notes = (
                    f"Devanagari script detected. Standard detector predicted '{stat_lang}'. "
                    f"Lacks distinctive regional markers; flagged for manual review."
                )

        # Case E: Latin script -> English or Transliterated Regional
        elif primary_script == "Latin":
            if lexical_matches:
                top_match, count = max(lexical_matches.items(), key=lambda x: x[1])
                primary_lang = top_match
                secondary_lang = "en"
                is_code_mixed = True
                confidence = 0.75
                method = "latin_transliteration_lexicon"
                notes = f"Latin script text containing transliterated {top_match} regional vocabulary."
            else:
                try:
                    primary_lang = langdetect.detect(raw_text)
                except Exception:
                    primary_lang = "en"
                confidence = 0.70
                method = "statistical_langdetect"
                notes = "Latin script identified via standard statistical model."

        else:
            # Fallback for unrecognized scripts
            try:
                primary_lang = langdetect.detect(raw_text)
                confidence = 0.50
            except Exception as e:
                primary_lang = "unknown"
                confidence = 0.0
            is_ambiguous = True
            review_recommended = True
            notes = "Unrecognized or mixed script. Low confidence."

        # Final confidence check against threshold
        if confidence < self.confidence_threshold:
            is_ambiguous = True
            review_recommended = True

        return LanguageIDResult(
            text=raw_text,
            primary_language=primary_lang,
            confidence=round(confidence, 3),
            secondary_language=secondary_lang,
            is_code_mixed=is_code_mixed,
            code_mixing_ratio=round(code_mix_ratio, 3),
            detected_scripts=detected_scripts,
            primary_script=primary_script,
            is_ambiguous=is_ambiguous,
            review_recommended=review_recommended,
            detection_method=method,
            notes=notes
        )

    def detect_batch(self, texts: List[str]) -> List[LanguageIDResult]:
        """Processes a list of texts through the language detector."""
        return [self.detect(t) for t in texts]

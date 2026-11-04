"""Text preprocessing pipeline preserving negation, emojis, slang, and dialect expressions."""

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Multilingual negation lexicon across all 7 regional languages + common contact languages
REGIONAL_NEGATION_TERMS: Set[str] = {
    # Tulu (Kannada script & Latin transliteration)
    "ಇಜ್ಜಿ", "ಇಜ್ಜ", "ಅತ್ತ", "ಅತ್ತ್", "ಬೊಡ್ಚಿ", "ಬೋಡ್ಚಿ", "ijji", "atta", "bodchi", "boddchi",
    # Assamese (Bengali-Assamese script & Latin)
    "নহয়", "নাই", "নালাগে", "নকৰোঁ", "নোৱাৰো", "নহব", "nohoy", "nai", "nalage", "nohobo",
    # Konkani (Devanagari & Latin)
    "ना", "न्हय", "नाका", "नुसत", "na", "nhay", "naka",
    # Santali (Ol Chiki & Latin)
    "ᱵᱟᱝ", "ᱵᱟ", "ᱵᱟᱭ", "ᱟᱞᱚ", "bang", "ba", "alo",
    # Bodo (Devanagari & Latin)
    "दा", "नङा", "गैया", "हाया", "da", "nonga", "gaiya", "haya",
    # Bhojpuri (Devanagari & Latin)
    "नाहीं", "ना", "मत", "नाइ", "नाहि", "nahin", "naahi", "mat",
    # Dogri (Devanagari & Latin)
    "नेईं", "ना", "मत", "नाहीं", "नेई", "nein", "nei",
    # Tamil (Tamil script & Latin)
    "இல்லை", "இல்ல", "மாட்டேன்", "கூடாது", "அல்ல", "வேண்டாம்", "illai", "illa", "maaten", "vendam",
    # Telugu (Telugu script & Latin)
    "లేదు", "కాదు", "వద్దు", "లేవు", "రాదు", "ledu", "kaadu", "vaddu", "raadu",
    # Malayalam (Malayalam script & Latin)
    "ഇല്ല", "അല്ല", "അരുത്", "വേണ്ട", "പാടില്ല", "illa", "alla", "venda", "aruthu",
    # Kannada (Kannada script & Latin)
    "ಇಲ್ಲ", "ಅಲ್ಲ", "ಬೇಡ", "ಬಾರದು", "ಸಾಕು", "illa", "alla", "beda", "baaradu",
    # Contact languages (Hindi / English loanwords)
    "not", "no", "never", "none", "neither", "nahi", "nahin", "nhi", "naa", "dont", "don't"
}

# Regex to detect individual emoji units in Unicode (including variation selectors)
EMOJI_PATTERN = re.compile(
    r"(\u2764\ufe0f|"
    r"[\U0001F600-\U0001F64F"  # emoticons
    r"\U0001F300-\U0001F5FF"  # symbols & pictographs
    r"\U0001F680-\U0001F6FF"  # transport & map
    r"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    r"\U00002702-\U000027B0"
    r"\U000024C2-\U0001F251"
    r"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    r"\U0001FA70-\U0001FAFF])",  # symbols and pictographs extended-a
    flags=re.UNICODE,
)

# Regex for sentiment-bearing punctuation
SENTIMENT_PUNCTUATION_PATTERN = re.compile(r"([!?]){2,}")


@dataclass
class PreprocessingResult:
    """Container holding original, processed text, and linguistic audit metadata."""
    original_text: str
    processed_text: str
    detected_negations: List[str] = field(default_factory=list)
    detected_emojis: List[str] = field(default_factory=list)
    detected_slang_tokens: List[str] = field(default_factory=list)
    is_transliterated_candidate: bool = False
    is_duplicate: bool = False
    duplicate_of_id: Optional[str] = None
    hash_sha256: str = ""
    warnings: List[str] = field(default_factory=list)


class TextPreprocessor:
    """Robust text preprocessor for low-resource Indian languages.
    
    Preserves original scripts, dialectal slang, emojis, negations, and sentiment markers.
    """

    def __init__(
        self,
        preserve_negations: bool = True,
        preserve_emojis: bool = True,
        preserve_slang: bool = True,
        normalize_unicode: bool = True,
        clean_excessive_whitespace: bool = True,
        near_duplicate_threshold: float = 0.88,
    ):
        self.preserve_negations = preserve_negations
        self.preserve_emojis = preserve_emojis
        self.preserve_slang = preserve_slang
        self.normalize_unicode = normalize_unicode
        self.clean_excessive_whitespace = clean_excessive_whitespace
        self.near_duplicate_threshold = near_duplicate_threshold

        # Cache for exact and near-duplicate detection
        self._seen_hashes: Dict[str, str] = {}  # sha256 -> sample_id
        self._seen_token_sets: List[Tuple[str, Set[str]]] = []  # (sample_id, set of ngrams)

    def reset_deduplication_cache(self) -> None:
        """Clears the duplicate detection cache."""
        self._seen_hashes.clear()
        self._seen_token_sets.clear()

    @staticmethod
    def normalize_text_unicode(text: str) -> str:
        """Applies Unicode NFC normalization without stripping native Indic script characters."""
        if not text:
            return ""
        # NFC preserves pre-composed characters while normalizing decomposed diacritics
        normalized = unicodedata.normalize("NFC", text)
        # Remove zero-width non-joiner / joiner corruptions while keeping valid joiners
        normalized = normalized.replace("\uFEFF", "")  # Byte order mark
        return normalized

    @staticmethod
    def extract_emojis(text: str) -> List[str]:
        """Finds all emojis present in the text."""
        return EMOJI_PATTERN.findall(text)

    @staticmethod
    def extract_negations(text: str) -> List[str]:
        """Detects presence of regional and contact negation markers."""
        clean_text = unicodedata.normalize("NFC", text).lower()
        punctuation_to_strip = ".,!?:;\"'()[]{}|॥।«»“”‘’"
        tokens = [tok.strip(punctuation_to_strip) for tok in clean_text.split() if tok.strip(punctuation_to_strip)]
        
        detected = []
        for tok in tokens:
            if tok in REGIONAL_NEGATION_TERMS:
                detected.append(tok)
                
        for term in REGIONAL_NEGATION_TERMS:
            if " " in term and term in clean_text and term not in detected:
                detected.append(term)
                
        return detected

    @staticmethod
    def is_likely_transliterated(text: str) -> bool:
        """Checks if text contains Latin-script transliterated regional Indian language."""
        if not text:
            return False
        # Calculate ratio of Latin characters
        latin_chars = len(re.findall(r"[a-zA-Z]", text))
        indic_chars = len(re.findall(r"[\u0900-\u0D7F\u1C50-\u1C7F]", text))
        total_letters = latin_chars + indic_chars

        if total_letters == 0:
            return False

        # If predominantly Latin letters but has distinct regional keywords
        if latin_chars / total_letters > 0.7:
            lower = text.lower()
            regional_latin_markers = [
                "edde", "laik", "ijji", "daane", "kudla", "miki", "tugi", "changa",
                "neek", "badhiya", "ka baat", "nohoy", "bhal", "borem", "sobit", "alo", "bang"
            ]
            for marker in regional_latin_markers:
                if marker in lower:
                    return True
        return False

    @staticmethod
    def _compute_char_ngrams(text: str, n: int = 3) -> Set[str]:
        """Generates character n-grams for fuzzy near-duplicate matching."""
        clean = re.sub(r"\s+", "", text.lower())
        if len(clean) < n:
            return {clean}
        return {clean[i : i + n] for i in range(len(clean) - n + 1)}

    def check_duplicate(self, text: str, sample_id: str) -> Tuple[bool, Optional[str], str]:
        """Checks for exact SHA-256 match or fuzzy character n-gram near-duplicates."""
        norm_text = re.sub(r"\s+", " ", text.strip().lower())
        text_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

        # Exact duplicate
        if text_hash in self._seen_hashes:
            return True, self._seen_hashes[text_hash], text_hash

        # Near-duplicate via Jaccard similarity of character 3-grams
        ngrams = self._compute_char_ngrams(norm_text, n=3)
        for prev_id, prev_ngrams in self._seen_token_sets:
            intersection = len(ngrams.intersection(prev_ngrams))
            union = len(ngrams.union(prev_ngrams))
            if union > 0:
                similarity = intersection / union
                if similarity >= self.near_duplicate_threshold:
                    return True, prev_id, text_hash

        # Register new entry
        self._seen_hashes[text_hash] = sample_id
        self._seen_token_sets.append((sample_id, ngrams))
        return False, None, text_hash

    def preprocess(
        self,
        text: str,
        sample_id: Optional[str] = None,
        check_duplicates: bool = True,
    ) -> PreprocessingResult:
        """Main preprocessing function for individual text samples.

        Args:
            text: Raw input text.
            sample_id: Optional ID for duplicate tracking.
            check_duplicates: Whether to evaluate for duplicates.

        Returns:
            PreprocessingResult: Preprocessed text and metadata.
        """
        sid = sample_id or f"hash_{abs(hash(text))}"
        raw = str(text) if text is not None else ""

        # Step 1: Handle Unicode normalization
        processed = self.normalize_text_unicode(raw) if self.normalize_unicode else raw

        # Step 2: Remove malformed control characters while preserving newlines & native scripts
        processed = "".join(
            ch for ch in processed
            if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", " ")
        )

        # Step 3: Extract & preserve emojis
        emojis = self.extract_emojis(processed) if self.preserve_emojis else []

        # Step 4: Extract & preserve negations
        negations = self.extract_negations(processed) if self.preserve_negations else []

        # Step 5: Normalize repeated sentiment punctuation (e.g. '!!!!!!' -> '!', '????' -> '?')
        # We preserve single or double punctuation as they bear high sentiment intensity
        processed = re.sub(r"!{2,}", "!", processed)
        processed = re.sub(r"\?{2,}", "?", processed)

        # Step 6: Handle excessive whitespace
        if self.clean_excessive_whitespace:
            processed = re.sub(r"[ \t]+", " ", processed).strip()

        # Step 7: Transliteration check
        is_trans = self.is_likely_transliterated(processed)

        # Step 8: Deduplication check
        is_dup = False
        dup_of = None
        text_hash = ""
        if check_duplicates and processed:
            is_dup, dup_of, text_hash = self.check_duplicate(processed, sid)

        result = PreprocessingResult(
            original_text=raw,
            processed_text=processed,
            detected_negations=negations,
            detected_emojis=emojis,
            is_transliterated_candidate=is_trans,
            is_duplicate=is_dup,
            duplicate_of_id=dup_of,
            hash_sha256=text_hash,
        )

        if not processed:
            result.warnings.append("Text is empty after preprocessing.")

        return result

    def preprocess_batch(
        self,
        texts: List[str],
        sample_ids: Optional[List[str]] = None,
        check_duplicates: bool = True,
    ) -> List[PreprocessingResult]:
        """Preprocesses a batch of text samples."""
        ids = sample_ids or [f"idx_{i}" for i in range(len(texts))]
        results = []
        for text, sid in zip(texts, ids):
            results.append(self.preprocess(text, sample_id=sid, check_duplicates=check_duplicates))
        return results

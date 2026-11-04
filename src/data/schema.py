"""Data schemas and type definitions for text and audio sentiment datasets."""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class EmotionLabel(str, Enum):
    """The 5 primary emotion categories for fine-grained emotional analysis."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    NEUTRAL = "neutral"


class SentimentLabel(str, Enum):
    """Emotion and sentiment label representations supporting 5 canonical emotions and legacy classes."""
    # 5 Canonical Emotion Classes
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    NEUTRAL = "neutral"
    # Legacy Sentiment Compatibility
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNLABELED = "unlabeled"

    def to_5_emotion(self) -> "SentimentLabel":
        """Maps any sentiment or emotion label into one of the 5 canonical emotions."""
        mapping = {
            SentimentLabel.POSITIVE: SentimentLabel.JOY,
            SentimentLabel.JOY: SentimentLabel.JOY,
            SentimentLabel.NEGATIVE: SentimentLabel.SADNESS,
            SentimentLabel.SADNESS: SentimentLabel.SADNESS,
            SentimentLabel.ANGER: SentimentLabel.ANGER,
            SentimentLabel.FEAR: SentimentLabel.FEAR,
            SentimentLabel.NEUTRAL: SentimentLabel.NEUTRAL,
        }
        return mapping.get(self, SentimentLabel.NEUTRAL)

    @classmethod
    def from_str(cls, val: Optional[str]) -> "SentimentLabel":
        if not val or str(val).strip().lower() in ("", "none", "nan", "unlabeled", "unknown"):
            return cls.UNLABELED
        val_clean = str(val).strip().lower()
        # Direct 5 emotions
        if val_clean in ("joy", "happy", "happiness", "joyful", "excited", "love", "delight"):
            return cls.JOY
        elif val_clean in ("sadness", "sad", "sorrow", "depressed", "unhappy", "grief"):
            return cls.SADNESS
        elif val_clean in ("anger", "angry", "rage", "furious", "annoyed", "irritated"):
            return cls.ANGER
        elif val_clean in ("fear", "scared", "afraid", "fearful", "anxiety", "worried", "panic"):
            return cls.FEAR
        elif val_clean in ("neutral", "neu", "0", "indifferent", "objective", "calm"):
            return cls.NEUTRAL
        # Compatibility sentiment mappings
        elif val_clean in ("positive", "pos", "+1", "1"):
            return cls.POSITIVE
        elif val_clean in ("negative", "neg", "-1"):
            return cls.NEGATIVE
        raise ValueError(
            f"Invalid sentiment/emotion label: '{val}'. "
            "Must be joy, sadness, anger, fear, neutral, positive, negative, or unlabeled."
        )



class LanguageStyle(str, Enum):
    DIRECT = "direct"
    BILINGUAL = "bilingual"
    CODE_MIXED = "code_mixed"
    SLANG_DIALECT_HEAVY = "slang_dialect_heavy"

    @classmethod
    def from_str(cls, val: Optional[str]) -> Optional["LanguageStyle"]:
        if not val or str(val).strip().lower() in ("", "none", "nan", "unlabeled", "unknown"):
            return None
        c = str(val).strip().lower().replace("-", "_").replace(" ", "_")
        if c in ("direct", "monolingual"):
            return cls.DIRECT
        elif c in ("bilingual", "bi"):
            return cls.BILINGUAL
        elif c in ("code_mixed", "codemixed", "code_switch", "codeswitch", "mixed"):
            return cls.CODE_MIXED
        elif "slang" in c or "dialect" in c or "heavy" in c:
            return cls.SLANG_DIALECT_HEAVY
        return cls.DIRECT


class InputType(str, Enum):
    DIRECT = "direct"
    BILINGUAL = "bilingual"
    CODE_MIXED = "code_mixed"
    TEXT = "text"
    AUDIO = "audio"

    @classmethod
    def from_str(cls, val: Optional[str]) -> "InputType":
        if not val or str(val).strip().lower() in ("", "none", "nan"):
            return cls.DIRECT
        val_clean = str(val).strip().lower()
        if val_clean in ("text", "txt"):
            return cls.TEXT
        elif val_clean in ("audio", "aud", "speech"):
            return cls.AUDIO
        elif val_clean in ("bilingual", "bi"):
            return cls.BILINGUAL
        elif val_clean in ("code_mixed", "codemixed", "code-mixed", "mixed"):
            return cls.CODE_MIXED
        elif val_clean == "direct":
            return cls.DIRECT
        raise ValueError(f"Invalid input_type: '{val}'. Must be 'text', 'audio', 'direct', 'bilingual', or 'code_mixed'.")


class TextRecord(BaseModel):
    """Schema for text sentiment samples."""
    sample_id: str = Field(..., description="Unique sample identifier")
    text: str = Field(..., description="Original raw text sample")
    language: str = Field(..., description="Primary language identifier or ISO code")
    secondary_language: Optional[str] = Field(None, description="Optional secondary language identifier")
    input_type: InputType = Field(default=InputType.DIRECT, description="text, audio, direct, bilingual, or code_mixed")
    language_style: Optional[LanguageStyle] = Field(None, description="Direct, Bilingual, Code-mixed, or Slang-dialect-heavy")
    audio_path: Optional[str] = Field(None, description="Optional recording path when applicable")
    sentiment: Optional[SentimentLabel] = Field(default=SentimentLabel.UNLABELED, description="Sentiment label if known")
    script: Optional[str] = Field(None, description="Script name, e.g. Devanagari, Kannada, Ol Chiki")
    dialect: Optional[str] = Field(None, description="Dialect or regional variation")
    source: Optional[str] = Field(None, description="Origin or dataset source name")

    # Optional processing outputs
    processed_text: Optional[str] = Field(None, description="Preprocessed text")
    language_confidence: Optional[float] = Field(None, description="Confidence score from language detector")
    is_ambiguous: Optional[bool] = Field(None, description="Flag indicating ambiguous language/script")

    model_config = {"extra": "ignore"}

    @field_validator("sample_id")
    def validate_sample_id(cls, v: str) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("sample_id cannot be empty")
        return s

    @field_validator("text")
    def validate_text(cls, v: str) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("text cannot be empty")
        return s

    @field_validator("language")
    def validate_language(cls, v: str) -> str:
        s = str(v).strip().lower()
        if not s:
            raise ValueError("language cannot be empty")
        return s

    @field_validator("sentiment", mode="before")
    def parse_sentiment(cls, v: Any) -> SentimentLabel:
        if isinstance(v, SentimentLabel):
            return v
        return SentimentLabel.from_str(v)

    @field_validator("input_type", mode="before")
    def parse_input_type(cls, v: Any) -> InputType:
        if isinstance(v, InputType):
            return v
        return InputType.from_str(v)

    @field_validator("language_style", mode="before")
    def parse_language_style(cls, v: Any) -> Optional[LanguageStyle]:
        if isinstance(v, LanguageStyle) or v is None:
            return v
        return LanguageStyle.from_str(v)


class AudioRecord(BaseModel):
    """Schema for audio recording samples and metadata."""
    sample_id: str = Field(..., description="Unique audio sample identifier")
    audio_path: str = Field(..., description="Path to the audio recording file")
    language: str = Field(..., description="Target or spoken language identifier")
    transcript: Optional[str] = Field(None, description="Transcription text if available")
    sentiment: Optional[SentimentLabel] = Field(default=SentimentLabel.UNLABELED, description="Sentiment label if annotated")
    dialect: Optional[str] = Field(None, description="Dialect variation")
    source: Optional[str] = Field(None, description="Origin or corpus source")

    # Audio technical attributes (populated upon inspection)
    duration_seconds: Optional[float] = Field(None, description="Audio duration in seconds")
    sample_rate: Optional[int] = Field(None, description="Native sampling rate")
    channels: Optional[int] = Field(None, description="Number of channels (1=mono, 2=stereo)")
    is_valid: Optional[bool] = Field(None, description="Whether the audio file is verified and readable")

    @field_validator("sample_id")
    def validate_sample_id(cls, v: str) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("sample_id cannot be empty")
        return s

    @field_validator("audio_path")
    def validate_audio_path(cls, v: str) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("audio_path cannot be empty")
        return s

    @field_validator("language")
    def validate_language(cls, v: str) -> str:
        s = str(v).strip().lower()
        if not s:
            raise ValueError("language cannot be empty")
        return s

    @field_validator("sentiment", mode="before")
    def parse_sentiment(cls, v: Any) -> SentimentLabel:
        if isinstance(v, SentimentLabel):
            return v
        return SentimentLabel.from_str(v)


class ValidationReport(BaseModel):
    """Comprehensive validation result summary."""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    duplicate_ids: List[str] = Field(default_factory=list)
    missing_fields_summary: Dict[str, int] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    unsupported_languages: List[str] = Field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.invalid_records == 0 and len(self.duplicate_ids) == 0

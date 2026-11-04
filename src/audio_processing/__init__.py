"""Audio processing module exports."""

from src.audio_processing.sraavaani_asr import (
    SraVaaniASRClient,
    TranscriptionResult,
    TranscriptionStatus,
)
from src.audio_processing.xlsr_features import (
    XLSRFeatureExtractor,
    AudioEmbeddingOutput,
)
from src.audio_processing.opensmile_features import (
    OpenSMILEFeatureExtractor,
    AcousticFeatureOutput,
)

__all__ = [
    "SraVaaniASRClient",
    "TranscriptionResult",
    "TranscriptionStatus",
    "XLSRFeatureExtractor",
    "AudioEmbeddingOutput",
    "OpenSMILEFeatureExtractor",
    "AcousticFeatureOutput",
]

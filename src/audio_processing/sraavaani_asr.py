"""SraVaani ASR integration interface for regional spoken audio transcription."""

import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Union
import requests

from src.data.schema import AudioRecord
from src.preprocessing.audio_preprocessing import AudioPreprocessor, AudioInspection
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class TranscriptionStatus(str, Enum):
    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    INVALID_AUDIO = "invalid_audio"


@dataclass
class TranscriptionResult:
    """Standardized result returned by SraVaani transcription interface."""
    sample_id: str
    audio_path: str
    language: str
    transcript: Optional[str] = None
    confidence: Optional[float] = None
    status: TranscriptionStatus = TranscriptionStatus.UNAVAILABLE
    error: Optional[str] = None
    processing_time_sec: float = 0.0
    model_id: str = "sraavaani-asr-multilingual"


class SraVaaniASRClient:
    """Client interface for SraVaani ASR (IISc / RESPIN initiative).
    
    Adheres strictly to the integrity constraint:
    - Performs real transcription if a valid endpoint and API key/model checkpoint are provided.
    - If unconfigured, returns TranscriptionStatus.UNAVAILABLE with clear diagnostic documentation.
    - NEVER fabricates fake transcriptions or substitutes unrelated models.
    """

    SUPPORTED_LANGUAGES = {"tulu", "tcy", "kannada", "kn"}

    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: int = 30,
        model_id: str = "sraavaani-asr-tulu",
    ):
        self.api_endpoint = api_endpoint or os.environ.get("SRAVAANI_API_ENDPOINT")
        self.api_key = api_key or os.environ.get("SRAVAANI_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.model_id = model_id
        self.audio_preprocessor = AudioPreprocessor()

    @property
    def is_service_configured(self) -> bool:
        """Checks whether the external SraVaani endpoint and credentials are configured."""
        return bool(self.api_endpoint and str(self.api_endpoint).strip())

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: str = "tulu",
        sample_id: Optional[str] = None,
        validate_audio: bool = True,
    ) -> TranscriptionResult:
        """Transcribes a regional audio recording using SraVaani ASR.

        Args:
            audio_path: Path to the audio recording file.
            language: Spoken language code (e.g. 'tulu', 'tcy').
            sample_id: Optional identifier for tracking.
            validate_audio: Whether to inspect audio integrity prior to request.

        Returns:
            TranscriptionResult: Transcript (if available), status, and latency.
        """
        start_time = time.time()
        path_str = str(audio_path)
        sid = sample_id or Path(path_str).stem
        lang_clean = language.strip().lower()

        # Step 1: Validate audio file readability
        if validate_audio:
            inspection: AudioInspection = self.audio_preprocessor.inspect_audio(path_str)
            if not inspection.is_valid:
                return TranscriptionResult(
                    sample_id=sid,
                    audio_path=path_str,
                    language=lang_clean,
                    transcript=None,
                    status=TranscriptionStatus.INVALID_AUDIO,
                    error=f"Audio validation failed: {inspection.error_message}",
                    processing_time_sec=round(time.time() - start_time, 3),
                    model_id=self.model_id,
                )

        # Step 2: Language capability check
        if lang_clean not in self.SUPPORTED_LANGUAGES:
            return TranscriptionResult(
                sample_id=sid,
                audio_path=path_str,
                language=lang_clean,
                transcript=None,
                status=TranscriptionStatus.UNSUPPORTED_LANGUAGE,
                error=(
                    f"Language '{language}' is not supported by SraVaani ASR. "
                    f"Supported languages: {sorted(list(self.SUPPORTED_LANGUAGES))}."
                ),
                processing_time_sec=round(time.time() - start_time, 3),
                model_id=self.model_id,
            )

        # Step 3: Check endpoint availability
        if not self.is_service_configured:
            logger.warning(
                f"SraVaani ASR endpoint not configured. Cannot transcribe '{path_str}'. "
                f"Configure SRAVAANI_API_ENDPOINT and SRAVAANI_API_KEY to enable live ASR."
            )
            return TranscriptionResult(
                sample_id=sid,
                audio_path=path_str,
                language=lang_clean,
                transcript=None,
                status=TranscriptionStatus.UNAVAILABLE,
                error=(
                    "SraVaani ASR service is currently unconfigured. "
                    "External API endpoint or local model checkpoint required. "
                    "Set SRAVAANI_API_ENDPOINT and SRAVAANI_API_KEY to enable live inference."
                ),
                processing_time_sec=round(time.time() - start_time, 3),
                model_id=self.model_id,
            )

        # Step 4: Dispatch real HTTP request to SraVaani API
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with open(path_str, "rb") as f:
                files = {"audio": (Path(path_str).name, f, "audio/wav")}
                data = {"language": lang_clean, "model": self.model_id}
                response = requests.post(
                    self.api_endpoint,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=self.timeout_seconds,
                )

            if response.status_code == 200:
                resp_json = response.json()
                transcript = resp_json.get("transcript", resp_json.get("text"))
                confidence = float(resp_json.get("confidence", 1.0))
                return TranscriptionResult(
                    sample_id=sid,
                    audio_path=path_str,
                    language=lang_clean,
                    transcript=transcript,
                    confidence=confidence,
                    status=TranscriptionStatus.SUCCESS,
                    processing_time_sec=round(time.time() - start_time, 3),
                    model_id=self.model_id,
                )
            else:
                return TranscriptionResult(
                    sample_id=sid,
                    audio_path=path_str,
                    language=lang_clean,
                    transcript=None,
                    status=TranscriptionStatus.FAILED,
                    error=f"SraVaani API returned status {response.status_code}: {response.text}",
                    processing_time_sec=round(time.time() - start_time, 3),
                    model_id=self.model_id,
                )
        except requests.RequestException as e:
            logger.error(f"Network error connecting to SraVaani endpoint {self.api_endpoint}: {e}")
            return TranscriptionResult(
                sample_id=sid,
                audio_path=path_str,
                language=lang_clean,
                transcript=None,
                status=TranscriptionStatus.FAILED,
                error=f"Connection error to SraVaani ASR: {str(e)}",
                processing_time_sec=round(time.time() - start_time, 3),
                model_id=self.model_id,
            )

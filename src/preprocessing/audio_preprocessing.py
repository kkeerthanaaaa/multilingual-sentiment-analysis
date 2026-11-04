"""Audio validation, conversion, noise filtering, and segmentation pipeline."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import soundfile as sf

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass
class AudioInspection:
    """Attributes and validation state of a raw audio recording."""
    file_path: str
    is_valid: bool
    format: str
    sample_rate: int = 0
    channels: int = 0
    duration_seconds: float = 0.0
    num_frames: int = 0
    error_message: Optional[str] = None
    is_silent: bool = False


@dataclass
class AudioSegment:
    """Metadata for an individual time window of a segmented recording."""
    segment_index: int
    start_time_seconds: float
    end_time_seconds: float
    output_path: str
    duration_seconds: float
    sample_rate: int


@dataclass
class AudioPreprocessingResult:
    """Output bundle of audio processing transformations."""
    original_path: str
    processed_path: str
    target_sample_rate: int
    target_channels: int
    duration_seconds: float
    was_resampled: bool
    was_normalized: bool
    applied_preemphasis: bool
    segments: List[AudioSegment] = field(default_factory=list)
    inspection: Optional[AudioInspection] = None
    error: Optional[str] = None


class AudioPreprocessor:
    """Robust audio validation and preprocessing pipeline.
    
    Supports format verification, duration checking, mono conversion, 16kHz resampling,
    DC offset removal, pre-emphasis filtering, RMS normalization, and windowed segmentation.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        min_duration_seconds: float = 0.3,
        max_duration_seconds: float = 30.0,
        segment_duration_seconds: float = 15.0,
        segment_overlap_seconds: float = 1.0,
        preemphasis_coef: float = 0.97,
        normalize_amplitude: bool = True,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds
        self.segment_duration_seconds = segment_duration_seconds
        self.segment_overlap_seconds = segment_overlap_seconds
        self.preemphasis_coef = preemphasis_coef
        self.normalize_amplitude = normalize_amplitude

    def inspect_audio(self, audio_path: str) -> AudioInspection:
        """Validates file existence, format readability, and duration."""
        p = Path(audio_path)
        ext = p.suffix.lower()
        if ext not in SUPPORTED_AUDIO_EXTENSIONS:
            return AudioInspection(
                file_path=str(audio_path),
                is_valid=False,
                format=ext or "unknown",
                error_message=f"Unsupported audio extension: {ext}. Supported: {SUPPORTED_AUDIO_EXTENSIONS}",
            )

        if not p.exists():
            return AudioInspection(
                file_path=str(audio_path),
                is_valid=False,
                format=ext,
                error_message=f"Audio file does not exist: {audio_path}",
            )

        if p.stat().st_size == 0:
            return AudioInspection(
                file_path=str(audio_path),
                is_valid=False,
                format=ext,
                error_message="Audio file is empty (0 bytes).",
            )

        try:
            info = sf.info(str(p))
            dur = float(info.duration)
            if dur < self.min_duration_seconds:
                return AudioInspection(
                    file_path=str(audio_path),
                    is_valid=False,
                    format=info.format,
                    sample_rate=info.samplerate,
                    channels=info.channels,
                    duration_seconds=dur,
                    num_frames=info.frames,
                    error_message=f"Audio duration {dur:.2f}s is below minimum threshold {self.min_duration_seconds}s.",
                )

            # Quick silence check
            data, _ = sf.read(str(p), frames=min(info.frames, 16000 * 5))
            rms = np.sqrt(np.mean(data**2))
            is_silent = rms < 1e-4

            return AudioInspection(
                file_path=str(audio_path),
                is_valid=True,
                format=info.format,
                sample_rate=info.samplerate,
                channels=info.channels,
                duration_seconds=round(dur, 3),
                num_frames=info.frames,
                is_silent=is_silent,
            )
        except Exception as e:
            return AudioInspection(
                file_path=str(audio_path),
                is_valid=False,
                format=ext,
                error_message=f"Failed to read audio header/stream: {str(e)}",
            )

    @staticmethod
    def _resample(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resamples 1D audio array from orig_sr to target_sr using polyphase filtering."""
        if orig_sr == target_sr:
            return audio_data
        num_target_samples = int(round(len(audio_data) * float(target_sr) / orig_sr))
        return signal.resample(audio_data, num_target_samples)

    @staticmethod
    def _apply_preemphasis(audio_data: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """Applies high-pass pre-emphasis filter to amplify high frequencies."""
        if coef <= 0.0:
            return audio_data
        return np.append(audio_data[0], audio_data[1:] - coef * audio_data[:-1])

    @staticmethod
    def _normalize_rms(audio_data: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Normalizes audio waveform to avoid clipping while maximizing dynamic range."""
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 1e-6:
            return audio_data * (target_peak / max_abs)
        return audio_data

    def segment_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        base_output_dir: str,
        base_filename: str,
    ) -> List[AudioSegment]:
        """Segments long audio into overlapping chunks."""
        total_samples = len(audio_data)
        chunk_samples = int(self.segment_duration_seconds * sample_rate)
        overlap_samples = int(self.segment_overlap_seconds * sample_rate)
        step_samples = max(1, chunk_samples - overlap_samples)

        os.makedirs(base_output_dir, exist_ok=True)
        segments: List[AudioSegment] = []
        idx = 0

        for start in range(0, total_samples, step_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = audio_data[start:end]
            dur = len(chunk) / sample_rate

            if dur < self.min_duration_seconds:
                continue

            seg_filename = f"{base_filename}_seg_{idx:03d}.wav"
            seg_path = os.path.join(base_output_dir, seg_filename)
            sf.write(seg_path, chunk.astype(np.float32), sample_rate, subtype="PCM_16")

            segments.append(
                AudioSegment(
                    segment_index=idx,
                    start_time_seconds=round(start / sample_rate, 3),
                    end_time_seconds=round(end / sample_rate, 3),
                    output_path=seg_path,
                    duration_seconds=round(dur, 3),
                    sample_rate=sample_rate,
                )
            )
            idx += 1
            if end >= total_samples:
                break

        return segments

    def preprocess_audio_file(
        self,
        audio_path: str,
        output_dir: str = "data/processed/audio",
        output_filename: Optional[str] = None,
        apply_preemphasis: bool = False,
        segment_if_long: bool = True,
    ) -> AudioPreprocessingResult:
        """Executes full validation, resampling, mono conversion, filtering, and segmentation.

        Args:
            audio_path: Path to raw input audio recording.
            output_dir: Directory where processed audio will be saved.
            output_filename: Optional custom output filename.
            apply_preemphasis: Whether to apply pre-emphasis filter.
            segment_if_long: Whether to segment recordings longer than max_duration_seconds.

        Returns:
            AudioPreprocessingResult: Status, processed audio path, and metadata.
        """
        inspection = self.inspect_audio(audio_path)
        if not inspection.is_valid:
            return AudioPreprocessingResult(
                original_path=audio_path,
                processed_path="",
                target_sample_rate=self.target_sample_rate,
                target_channels=self.target_channels,
                duration_seconds=0.0,
                was_resampled=False,
                was_normalized=False,
                applied_preemphasis=False,
                inspection=inspection,
                error=inspection.error_message,
            )

        os.makedirs(output_dir, exist_ok=True)
        orig_p = Path(audio_path)
        out_name = output_filename or f"proc_{orig_p.stem}.wav"
        out_path = os.path.join(output_dir, out_name)

        try:
            # Step 1: Read audio data
            data, sr = sf.read(str(orig_p), dtype="float32")

            # Step 2: Convert to mono if multi-channel
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            # Step 3: Remove DC offset
            data = data - np.mean(data)

            # Step 4: Resample if needed
            was_resampled = False
            if sr != self.target_sample_rate:
                data = self._resample(data, orig_sr=sr, target_sr=self.target_sample_rate)
                was_resampled = True

            # Step 5: Optional pre-emphasis
            did_preemphasis = False
            if apply_preemphasis:
                data = self._apply_preemphasis(data, coef=self.preemphasis_coef)
                did_preemphasis = True

            # Step 6: Normalization
            was_normalized = False
            if self.normalize_amplitude:
                data = self._normalize_rms(data, target_peak=0.95)
                was_normalized = True

            # Step 7: Write standardized 16kHz mono 16-bit PCM WAV
            sf.write(out_path, data.astype(np.float32), self.target_sample_rate, subtype="PCM_16")
            final_duration = len(data) / self.target_sample_rate

            # Step 8: Segment if recording is longer than max_duration_seconds
            segments = []
            if segment_if_long and final_duration > self.max_duration_seconds:
                seg_dir = os.path.join(output_dir, f"{orig_p.stem}_segments")
                segments = self.segment_audio(
                    audio_data=data,
                    sample_rate=self.target_sample_rate,
                    base_output_dir=seg_dir,
                    base_filename=orig_p.stem,
                )

            return AudioPreprocessingResult(
                original_path=str(audio_path),
                processed_path=out_path,
                target_sample_rate=self.target_sample_rate,
                target_channels=self.target_channels,
                duration_seconds=round(final_duration, 3),
                was_resampled=was_resampled,
                was_normalized=was_normalized,
                applied_preemphasis=did_preemphasis,
                segments=segments,
                inspection=inspection,
            )
        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}")
            return AudioPreprocessingResult(
                original_path=str(audio_path),
                processed_path="",
                target_sample_rate=self.target_sample_rate,
                target_channels=self.target_channels,
                duration_seconds=0.0,
                was_resampled=False,
                was_normalized=False,
                applied_preemphasis=False,
                inspection=inspection,
                error=f"Processing exception: {str(e)}",
            )

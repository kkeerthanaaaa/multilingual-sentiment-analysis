"""Acoustic feature extraction using openSMILE and signal processing fallbacks."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Check opensmile availability
try:
    import opensmile
    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False


@dataclass
class AcousticFeatureOutput:
    """Container for extracted low-level acoustic descriptors and prosody."""
    extractor_name: str
    feature_set: str
    feature_names: List[str]
    features_df: pd.DataFrame
    num_samples: int
    sample_ids: List[str] = field(default_factory=list)
    extraction_config: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


class OpenSMILEFeatureExtractor:
    """Acoustic feature extractor leveraging openSMILE (eGeMAPSv02 / ComParE_2016).
    
    Extracts fundamental frequency (F0/pitch), loudness, jitter, shimmer, HNR,
    formants, and spectral parameters. Includes a standalone NumPy/SciPy fallback.
    """

    SUPPORTED_FEATURE_SETS = {
        "eGeMAPSv02": "Extended Geneva Minimalistic Acoustic Parameter Set (88 features)",
        "GeMAPSv01b": "Geneva Minimalistic Acoustic Parameter Set (62 features)",
        "ComParE_2016": "Computational Paralinguistics Challenge Feature Set (6373 features)",
    }

    def __init__(
        self,
        feature_set: str = "eGeMAPSv02",
        feature_level: str = "Functionals",  # or 'LowLevelDescriptors'
        min_duration_seconds: float = 0.3,
    ):
        self.feature_set_name = feature_set
        self.feature_level_name = feature_level
        self.min_duration_seconds = min_duration_seconds

        self._smile = None
        if OPENSMILE_AVAILABLE:
            try:
                f_set = getattr(opensmile.FeatureSet, feature_set, opensmile.FeatureSet.eGeMAPSv02)
                f_level = getattr(opensmile.FeatureLevel, feature_level, opensmile.FeatureLevel.Functionals)
                self._smile = opensmile.Smile(
                    feature_set=f_set,
                    feature_level=f_level,
                )
            except Exception as e:
                logger.warning(f"Could not initialize openSMILE Smile object: {e}. Will use fallback.")
                self._smile = None

    @property
    def is_opensmile_active(self) -> bool:
        return self._smile is not None

    def _extract_fallback_acoustics(self, data: np.ndarray, sr: int) -> Dict[str, float]:
        """Pure NumPy/SciPy acoustic feature extraction for pitch, energy, and spectrum."""
        # 1. RMS Energy
        rms = float(np.sqrt(np.mean(data**2)))

        # 2. Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0) / len(data)

        # 3. Autocorrelation Pitch (F0) estimate
        corr = signal.correlate(data, data, mode="full")
        corr = corr[len(corr) // 2 :]
        # Search range 50 Hz to 450 Hz
        min_lag = int(sr / 450)
        max_lag = int(sr / 50)
        if len(corr) > max_lag and max_lag > min_lag:
            peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
            f0_estimate = float(sr / peak_lag) if peak_lag > 0 else 0.0
        else:
            f0_estimate = 0.0

        # 4. Spectral Centroid
        freqs, psd = signal.welch(data, sr, nperseg=min(len(data), 1024))
        centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-9))

        # 5. Spectral Rolloff (85% energy)
        cum_psd = np.cumsum(psd)
        rolloff_idx = np.where(cum_psd >= 0.85 * cum_psd[-1])[0]
        rolloff = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else 0.0

        return {
            "F0_estimate_Hz": round(f0_estimate, 2),
            "RMS_energy": round(rms, 6),
            "zero_crossing_rate": round(float(zero_crossings), 6),
            "spectral_centroid_Hz": round(centroid, 2),
            "spectral_rolloff_85_Hz": round(rolloff, 2),
            "duration_seconds": round(len(data) / sr, 3),
        }

    def extract_single_audio(
        self,
        audio_path: Union[str, Path],
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Extracts acoustic features for a single audio file.

        Args:
            audio_path: Path to audio recording.

        Returns:
            Tuple of (DataFrame with 1 row of features or None, error string or None).
        """
        p = Path(audio_path)
        if not p.exists():
            return None, f"File does not exist: {audio_path}"

        try:
            # Check duration and format
            info = sf.info(str(p))
            if info.duration < self.min_duration_seconds:
                return None, f"Audio duration {info.duration:.2f}s is below minimum {self.min_duration_seconds}s."

            # Primary: openSMILE
            if self._smile is not None:
                try:
                    df = self._smile.process_file(str(p))
                    return df, None
                except Exception as smile_err:
                    logger.warning(f"openSMILE processing failed for {p.name}: {smile_err}. Falling back to internal acoustic extractor.")

            # Fallback: NumPy/SciPy acoustic calculation
            data, sr = sf.read(str(p), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            feats = self._extract_fallback_acoustics(data, sr)
            df = pd.DataFrame([feats])
            return df, None

        except Exception as e:
            return None, f"Acoustic extraction error for {audio_path}: {str(e)}"

    def extract_batch(
        self,
        audio_paths: List[Union[str, Path]],
        sample_ids: Optional[List[str]] = None,
    ) -> AcousticFeatureOutput:
        """Extracts acoustic features across a batch of audio files."""
        sids = sample_ids or [Path(p).stem for p in audio_paths]
        dfs: List[pd.DataFrame] = []
        valid_sids: List[str] = []
        errors: Dict[str, str] = {}

        for p, sid in zip(audio_paths, sids):
            df, err = self.extract_single_audio(p)
            if df is not None and not df.empty:
                df = df.copy()
                df.insert(0, "sample_id", sid)
                dfs.append(df)
                valid_sids.append(sid)
            else:
                errors[sid] = err or "Acoustic extraction failure"

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            feature_names = [c for c in combined_df.columns if c != "sample_id"]
        else:
            combined_df = pd.DataFrame()
            feature_names = []

        extractor_name = "openSMILE" if self._smile is not None else "internal_scipy_acoustic"

        return AcousticFeatureOutput(
            extractor_name=extractor_name,
            feature_set=self.feature_set_name,
            feature_names=feature_names,
            features_df=combined_df,
            num_samples=len(valid_sids),
            sample_ids=valid_sids,
            extraction_config={
                "feature_set": self.feature_set_name,
                "feature_level": self.feature_level_name,
                "engine": extractor_name,
            },
            errors=errors,
        )

    @staticmethod
    def save_features(output: AcousticFeatureOutput, output_path: str) -> str:
        """Saves acoustic features DataFrame to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        output.features_df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Saved {output.num_samples} acoustic feature records to {output_path}")
        return output_path

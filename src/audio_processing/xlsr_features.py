"""XLS-R learned speech representation pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModel

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class AudioEmbeddingOutput:
    """Container for learned speech embeddings extracted via XLS-R."""
    model_name: str
    embeddings: np.ndarray  # Shape: (N, hidden_dim)
    pooling_method: str
    sample_rate: int
    hidden_dimension: int
    num_samples: int
    sample_ids: List[str] = field(default_factory=list)
    extraction_config: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


class XLSRFeatureExtractor:
    """Extracts dense continuous representations of speech using Wav2Vec2 / XLS-R.
    
    Default model: 'facebook/wav2vec2-xls-r-300m' (53-language cross-lingual speech model).
    Keeps learned embeddings strictly decoupled from acoustic features.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        target_sample_rate: int = 16000,
        pooling_method: str = "mean",
        device: Optional[str] = None,
        min_duration_seconds: float = 0.2,
    ):
        self.model_name = model_name
        self.target_sample_rate = target_sample_rate
        self.pooling_method = pooling_method.lower()
        self.min_duration_seconds = min_duration_seconds

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = None
        self.model = None

    def load_model(self) -> None:
        """Loads processor and model weights on demand."""
        if self.processor is None or self.model is None:
            logger.info(f"Loading XLS-R model '{self.model_name}' on {self.device}...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def _pool_speech_frames(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pools frame-level representations across time dimension (dim=1)."""
        if self.pooling_method == "max":
            return torch.max(hidden_states, dim=1)[0]
        else:  # 'mean'
            return torch.mean(hidden_states, dim=1)

    def extract_single_audio(
        self,
        audio_path: Union[str, Path],
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Extracts pooled embedding for a single audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            Tuple of (1D numpy array of shape (hidden_dim,) or None, error message or None).
        """
        p = Path(audio_path)
        if not p.exists():
            return None, f"File not found: {audio_path}"

        try:
            data, sr = sf.read(str(p), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            dur = len(data) / sr
            if dur < self.min_duration_seconds:
                return None, f"Audio duration {dur:.2f}s is below minimum {self.min_duration_seconds}s for XLS-R."

            # If sample rate differs, resample
            if sr != self.target_sample_rate:
                from scipy import signal
                num_target = int(round(len(data) * float(self.target_sample_rate) / sr))
                data = signal.resample(data, num_target)

            self.load_model()

            inputs = self.processor(
                data,
                sampling_rate=self.target_sample_rate,
                return_tensors="pt"
            )
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_values)
                last_hidden = outputs.last_hidden_state  # (1, T, hidden_dim)
                pooled = self._pool_speech_frames(last_hidden)  # (1, hidden_dim)
                emb = pooled.squeeze(0).cpu().numpy()

            return emb, None

        except Exception as e:
            return None, f"Extraction failed for {audio_path}: {str(e)}"

    def extract_batch(
        self,
        audio_paths: List[Union[str, Path]],
        sample_ids: Optional[List[str]] = None,
    ) -> AudioEmbeddingOutput:
        """Extracts learned embeddings for a list of audio files."""
        sids = sample_ids or [Path(p).stem for p in audio_paths]
        valid_embeddings: List[np.ndarray] = []
        valid_sids: List[str] = []
        errors: Dict[str, str] = {}

        for p, sid in zip(audio_paths, sids):
            emb, err = self.extract_single_audio(p)
            if emb is not None:
                valid_embeddings.append(emb)
                valid_sids.append(sid)
            else:
                errors[sid] = err or "Unknown extraction failure"

        if valid_embeddings:
            matrix = np.vstack(valid_embeddings)
            hidden_dim = matrix.shape[1]
        else:
            matrix = np.empty((0, 1024))
            hidden_dim = 1024

        config_record = {
            "model_name": self.model_name,
            "pooling_method": self.pooling_method,
            "target_sample_rate": self.target_sample_rate,
            "device": str(self.device),
        }

        return AudioEmbeddingOutput(
            model_name=self.model_name,
            embeddings=matrix,
            pooling_method=self.pooling_method,
            sample_rate=self.target_sample_rate,
            hidden_dimension=hidden_dim,
            num_samples=len(valid_embeddings),
            sample_ids=valid_sids,
            extraction_config=config_record,
            errors=errors,
        )

    @staticmethod
    def save_embeddings(output: AudioEmbeddingOutput, output_path: str) -> str:
        """Saves extracted embeddings and metadata to a compressed .npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez_compressed(
            output_path,
            embeddings=output.embeddings,
            sample_ids=np.array(output.sample_ids),
            hidden_dimension=output.hidden_dimension,
            model_name=output.model_name,
            pooling_method=output.pooling_method,
        )
        logger.info(f"Saved {output.num_samples} audio embeddings to {output_path}")
        return output_path

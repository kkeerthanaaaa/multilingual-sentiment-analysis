"""Configuration loader and manager for the pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class PathConfig:
    raw_text_dir: str = "data/raw/text"
    raw_audio_dir: str = "data/raw/audio"
    processed_dir: str = "data/processed"
    metadata_dir: str = "data/metadata"
    output_dir: str = "outputs"


@dataclass
class TextModelConfig:
    model_name: str = "xlm-roberta-base"
    max_seq_length: int = 128
    batch_size: int = 16
    device: str = "cpu"
    pooling_method: str = "mean"  # 'mean', 'cls', or 'max'
    indicbert_model_name: str = "ai4bharat/IndicBERTv2-MLM-only"


@dataclass
class AudioConfig:
    target_sample_rate: int = 16000
    target_channels: int = 1  # Mono
    max_duration_seconds: float = 30.0
    min_duration_seconds: float = 0.3
    segment_duration_seconds: float = 15.0
    segment_overlap_seconds: float = 1.0
    preemphasis_coef: float = 0.97
    normalize_amplitude: bool = True
    xlsr_model_name: str = "facebook/wav2vec2-xls-r-300m"
    opensmile_feature_set: str = "eGeMAPSv02"  # or 'ComParE_2016'


@dataclass
class SraVaaniConfig:
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: int = 30
    model_id: str = "sraavaani-asr-multilingual"


@dataclass
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    text_model: TextModelConfig = field(default_factory=TextModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    sraavaani: SraVaaniConfig = field(default_factory=SraVaaniConfig)
    languages: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AppConfig":
        """Loads configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML file.

        Returns:
            AppConfig: Loaded configuration instance.
        """
        p = Path(yaml_path)
        if not p.exists():
            logger.warning(f"Configuration file {yaml_path} not found. Using defaults.")
            return cls()

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            paths_data = data.get("paths", {})
            text_data = data.get("text_model", {})
            audio_data = data.get("audio", {})
            sraavaani_data = data.get("sraavaani", {})
            languages = data.get("languages", {})

            return cls(
                paths=PathConfig(**paths_data),
                text_model=TextModelConfig(**text_data),
                audio=AudioConfig(**audio_data),
                sraavaani=SraVaaniConfig(**sraavaani_data),
                languages=languages,
            )
        except Exception as e:
            logger.error(f"Error loading configuration from {yaml_path}: {e}")
            raise

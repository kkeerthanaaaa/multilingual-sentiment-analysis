"""Unified processing interface for Modules 1 & 2.

Exposes standard, reusable functions connecting dataset loading, text preprocessing,
language identification, audio inspection, conversion, transcription, and feature extraction.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import pandas as pd

from src.data.schema import TextRecord, AudioRecord, ValidationReport
from src.data.dataset_loader import DatasetLoader
from src.preprocessing.text_preprocessing import TextPreprocessor, PreprocessingResult
from src.preprocessing.language_detection import LanguageDetector, LanguageIDResult
from src.preprocessing.audio_preprocessing import (
    AudioPreprocessor,
    AudioInspection,
    AudioPreprocessingResult,
)
from src.text_processing.xlm_roberta import XLMRobertaProcessor, TextEmbeddingOutput
from src.text_processing.indicbert import IndicBERTProcessor
from src.text_processing.muril import MuRILProcessor
from src.audio_processing.sraavaani_asr import SraVaaniASRClient, TranscriptionResult
from src.audio_processing.xlsr_features import XLSRFeatureExtractor, AudioEmbeddingOutput
from src.audio_processing.opensmile_features import OpenSMILEFeatureExtractor, AcousticFeatureOutput
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Global reusable singletons for default lightweight usage
_DEFAULT_LOADER = DatasetLoader()
_DEFAULT_TEXT_PREPROCESSOR = TextPreprocessor()
_DEFAULT_LANG_DETECTOR = LanguageDetector()
_DEFAULT_AUDIO_PREPROCESSOR = AudioPreprocessor()
_DEFAULT_ASR_CLIENT = SraVaaniASRClient()


def load_text_dataset(
    file_path: Union[str, Path],
    file_format: Optional[str] = None,
    validate: bool = True,
) -> Tuple[pd.DataFrame, List[TextRecord], Optional[ValidationReport]]:
    """Loads a text sentiment dataset from a CSV or JSONL file.

    Args:
        file_path: Path to dataset file.
        file_format: Optional format string ('csv', 'jsonl').
        validate: Whether to validate rows against TextRecord schema.

    Returns:
        Tuple of (DataFrame, list of validated TextRecords, ValidationReport or None).
    """
    return _DEFAULT_LOADER.load_text_dataset(
        file_path=file_path, file_format=file_format, validate=validate
    )


def preprocess_text(
    text: str,
    sample_id: Optional[str] = None,
    check_duplicates: bool = True,
    preserve_negations: bool = True,
    preserve_emojis: bool = True,
) -> PreprocessingResult:
    """Preprocesses a single text sample while preserving scripts, slang, emojis, and negations.

    Args:
        text: Raw text string.
        sample_id: Optional ID for duplicate tracking.
        check_duplicates: Whether to check against seen text hashes.
        preserve_negations: Whether to protect negation tokens.
        preserve_emojis: Whether to protect emojis.

    Returns:
        PreprocessingResult: Cleaned text, extracted features, and audit metadata.
    """
    preprocessor = TextPreprocessor(
        preserve_negations=preserve_negations,
        preserve_emojis=preserve_emojis,
    )
    return preprocessor.preprocess(text, sample_id=sample_id, check_duplicates=check_duplicates)


def identify_language(
    text: str,
    confidence_threshold: float = 0.65,
) -> LanguageIDResult:
    """Identifies the primary regional language, script, confidence, and ambiguity flag.

    Args:
        text: Input text sample.
        confidence_threshold: Minimum confidence score before flagging ambiguity.

    Returns:
        LanguageIDResult: Structured language identification and audit notes.
    """
    detector = LanguageDetector(confidence_threshold=confidence_threshold)
    return detector.detect(text)


def detect_code_mixing(
    text: str,
    code_mixing_threshold: float = 0.20,
) -> Tuple[bool, float, Optional[str]]:
    """Detects whether an input text exhibits bilingual or code-mixed patterns.

    Args:
        text: Input text sample.
        code_mixing_threshold: Minimum script ratio to trigger code-mixing flag.

    Returns:
        Tuple of (is_code_mixed, code_mixing_ratio, secondary_language_code).
    """
    detector = LanguageDetector(code_mixing_threshold=code_mixing_threshold)
    res = detector.detect(text)
    return res.is_code_mixed, res.code_mixing_ratio, res.secondary_language


def validate_audio(audio_path: Union[str, Path]) -> AudioInspection:
    """Inspects audio file readability, format, duration, and channel configuration.

    Args:
        audio_path: Path to the recording file.

    Returns:
        AudioInspection: Validation status, format, duration, and error diagnostics.
    """
    return _DEFAULT_AUDIO_PREPROCESSOR.inspect_audio(str(audio_path))


def preprocess_audio(
    audio_path: Union[str, Path],
    output_dir: str = "data/processed/audio",
    target_sample_rate: int = 16000,
    apply_preemphasis: bool = False,
    segment_if_long: bool = True,
) -> AudioPreprocessingResult:
    """Validates, converts to 16kHz mono WAV, normalizes, and segments long audio.

    Args:
        audio_path: Path to raw input recording.
        output_dir: Output directory for processed WAVs.
        target_sample_rate: Standard sampling rate (default: 16000 Hz).
        apply_preemphasis: Whether to apply high-pass pre-emphasis filter.
        segment_if_long: Whether to chunk recordings longer than max duration.

    Returns:
        AudioPreprocessingResult: Processed audio path, segments list, and inspection.
    """
    processor = AudioPreprocessor(target_sample_rate=target_sample_rate)
    return processor.preprocess_audio_file(
        audio_path=str(audio_path),
        output_dir=output_dir,
        apply_preemphasis=apply_preemphasis,
        segment_if_long=segment_if_long,
    )


def transcribe_audio(
    audio_path: Union[str, Path],
    language: str = "tulu",
    client: Optional[SraVaaniASRClient] = None,
) -> TranscriptionResult:
    """Transcribes a regional audio recording using SraVaani ASR.
    
    Returns explicit UNAVAILABLE status when external endpoints are unconfigured;
    never fabricates fake transcription text.

    Args:
        audio_path: Path to audio file.
        language: Target spoken language code (e.g. 'tulu', 'tcy').
        client: Optional custom SraVaaniASRClient instance.

    Returns:
        TranscriptionResult: Transcript (if available), status, and latency.
    """
    c = client or _DEFAULT_ASR_CLIENT
    return c.transcribe(audio_path=audio_path, language=language)


def extract_text_embeddings(
    texts: List[str],
    model_type: str = "xlm-roberta",
    model_name: Optional[str] = None,
    pooling_method: str = "mean",
    batch_size: int = 16,
) -> TextEmbeddingOutput:
    """Extracts dense sentence embeddings using XLM-RoBERTa or IndicBERT.

    Args:
        texts: List of text strings.
        model_type: 'xlm-roberta' or 'indicbert'.
        model_name: Optional Hugging Face model identifier to override default.
        pooling_method: 'mean', 'cls', or 'max'.
        batch_size: Inference batch size.

    Returns:
        TextEmbeddingOutput: Numpy embeddings matrix of shape (N, hidden_dim).
    """
    m_type = model_type.strip().lower()
    if m_type in ("xlm-roberta", "xlm_roberta", "xlmr"):
        name = model_name or "xlm-roberta-base"
        proc = XLMRobertaProcessor(
            model_name=name, pooling_method=pooling_method, batch_size=batch_size
        )
        return proc.extract_embeddings(texts)
    elif m_type in ("indicbert", "indic_bert"):
        name = model_name or "ai4bharat/IndicBERTv2-MLM-only"
        proc = IndicBERTProcessor(
            model_name=name, pooling_method=pooling_method, batch_size=batch_size
        )
        return proc.extract_embeddings(texts)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'xlm-roberta' or 'indicbert'.")


def extract_audio_embeddings(
    audio_paths: List[Union[str, Path]],
    model_name: str = "facebook/wav2vec2-xls-r-300m",
    pooling_method: str = "mean",
) -> AudioEmbeddingOutput:
    """Extracts learned speech representations using XLS-R.

    Args:
        audio_paths: List of audio file paths.
        model_name: Pretrained Wav2Vec2/XLS-R checkpoint name.
        pooling_method: 'mean' or 'max' across time steps.

    Returns:
        AudioEmbeddingOutput: Matrix of dense audio embeddings and metadata.
    """
    extractor = XLSRFeatureExtractor(model_name=model_name, pooling_method=pooling_method)
    return extractor.extract_batch(audio_paths)


def extract_acoustic_features(
    audio_paths: List[Union[str, Path]],
    feature_set: str = "eGeMAPSv02",
) -> AcousticFeatureOutput:
    """Extracts acoustic prosodic parameters (pitch/F0, loudness, jitter, shimmer) via openSMILE.

    Args:
        audio_paths: List of audio file paths.
        feature_set: openSMILE feature set name ('eGeMAPSv02', 'ComParE_2016').

    Returns:
        AcousticFeatureOutput: DataFrame of acoustic parameters.
    """
    extractor = OpenSMILEFeatureExtractor(feature_set=feature_set)
    return extractor.extract_batch(audio_paths)


def fetch_real_dataset(
    output_dir: str = "data/raw/text",
    max_samples_per_lang: int = 200,
) -> Tuple[pd.DataFrame, Path]:
    """Fetches real-time benchmark datasets and builds standardized corpus."""
    from src.data.real_dataset_loader import RealDatasetFetcher
    fetcher = RealDatasetFetcher(output_dir=output_dir)
    return fetcher.build_real_time_dataset(max_samples_per_lang=max_samples_per_lang)


def train_sentiment(
    dataset_path: Optional[str] = None,
    model_type: str = "hybrid",
    epochs: int = 3,
    test_size: float = 0.2,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Trains a sentiment model on real-time multilingual regional language datasets."""
    from src.training.train_sentiment import train_sentiment_model
    return train_sentiment_model(
        dataset_path=dataset_path,
        model_type=model_type,
        epochs=epochs,
        test_size=test_size,
        output_dir=output_dir,
    )


def predict_sentiment_hybrid(
    texts: List[str],
    model_dir: str = "models/hybrid_sentiment",
    sample_ids: Optional[List[str]] = None,
) -> List[Any]:
    """Runs inference using the trained high-accuracy hybrid ensemble emotion classifier."""
    from src.text_processing.hybrid_classifier import HybridSentimentClassifier
    classifier = HybridSentimentClassifier()
    classifier.load(model_dir)
    return classifier.predict(texts=texts, sample_ids=sample_ids)


# Convenient alias for 5-emotion prediction
predict_emotion_hybrid = predict_sentiment_hybrid


def predict_emotion_muril(
    texts: List[str],
    sample_ids: Optional[List[str]] = None,
) -> List[Any]:
    """Runs inference using the MuRIL Indian language emotion classification pipeline."""
    processor = MuRILProcessor()
    return processor.predict_sentiment(texts=texts, sample_ids=sample_ids)



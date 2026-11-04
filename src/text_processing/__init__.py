"""Text processing module exports."""

from src.text_processing.xlm_roberta import (
    XLMRobertaProcessor,
    TextEmbeddingOutput,
    TextClassificationOutput,
    SENTIMENT_LABEL_MAP,
    ID2LABEL,
    LABEL2ID,
)
from src.text_processing.indicbert import (
    IndicBERTProcessor,
    IndicBERTLanguageCoverage,
    INDICBERT_V1_LANGUAGES,
    INDICBERT_V2_LANGUAGES,
)
from src.text_processing.muril import (
    MuRILProcessor,
    MuRILLanguageCoverage,
    MURIL_SUPPORTED_LANGUAGES,
)

__all__ = [
    "XLMRobertaProcessor",
    "TextEmbeddingOutput",
    "TextClassificationOutput",
    "SENTIMENT_LABEL_MAP",
    "ID2LABEL",
    "LABEL2ID",
    "IndicBERTProcessor",
    "IndicBERTLanguageCoverage",
    "INDICBERT_V1_LANGUAGES",
    "INDICBERT_V2_LANGUAGES",
    "MuRILProcessor",
    "MuRILLanguageCoverage",
    "MURIL_SUPPORTED_LANGUAGES",
]


"""Multilingual text processing and representation using XLM-RoBERTa."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from src.data.schema import TextRecord, SentimentLabel
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

SENTIMENT_LABEL_MAP: Dict[SentimentLabel, int] = {
    SentimentLabel.JOY: 0,
    SentimentLabel.SADNESS: 1,
    SentimentLabel.ANGER: 2,
    SentimentLabel.FEAR: 3,
    SentimentLabel.NEUTRAL: 4,
    SentimentLabel.POSITIVE: 0,
    SentimentLabel.NEGATIVE: 1,
}
ID2LABEL: Dict[int, str] = {0: "joy", 1: "sadness", 2: "anger", 3: "fear", 4: "neutral"}
LABEL2ID: Dict[str, int] = {
    "joy": 0, "sadness": 1, "anger": 2, "fear": 3, "neutral": 4,
    "positive": 0, "negative": 1,
}


@dataclass
class TextEmbeddingOutput:
    """Container for extracted text embeddings."""
    model_name: str
    embeddings: np.ndarray  # Shape: (N, hidden_dim)
    pooling_method: str
    hidden_dimension: int
    num_samples: int
    sample_ids: List[str] = field(default_factory=list)


@dataclass
class TextClassificationOutput:
    """Container for sentiment classification predictions."""
    sample_id: str
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    model_trained: bool
    status_note: str


class XLMRobertaProcessor:
    """XLM-RoBERTa baseline pipeline for multilingual representation and classification.
    
    Provides tokenization, embedding pooling (mean, cls, max), batch extraction,
    supervised classification head, checkpoint persistence, and language coverage reporting.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_seq_length: int = 128,
        batch_size: int = 16,
        device: Optional[str] = None,
        pooling_method: str = "mean",
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.pooling_method = pooling_method.lower()

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.encoder = None
        self.classifier_model = None
        self.is_classifier_trained = False

    def load_encoder(self) -> None:
        """Loads XLM-R tokenizer and base transformer encoder lazily."""
        if self.tokenizer is None or self.encoder is None:
            logger.info(f"Loading XLM-RoBERTa encoder '{self.model_name}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.encoder = AutoModel.from_pretrained(self.model_name)
            self.encoder.to(self.device)
            self.encoder.eval()

    def load_classifier(self, num_labels: int = 5) -> None:
        """Loads sequence classification model."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.classifier_model is None:
            logger.info(f"Loading XLM-RoBERTa sequence classifier ({num_labels} labels) on {self.device}...")
            id2l = ID2LABEL if num_labels == 5 else {0: "negative", 1: "neutral", 2: "positive"}
            l2id = LABEL2ID if num_labels == 5 else {"negative": 0, "neutral": 1, "positive": 2}
            self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                id2label=id2l,
                label2id=l2id,
            )
            self.classifier_model.to(self.device)

    @staticmethod
    def get_language_coverage_notice(language: str) -> str:
        """Factual coverage report for XLM-RoBERTa on target regional languages."""
        lang_lower = language.lower()
        if lang_lower in ("tamil", "ta", "tam"):
            return "XLM-R officially covers Tamil (Tamil script) in its 100-language pretraining set."
        elif lang_lower in ("telugu", "te", "tel"):
            return "XLM-R officially covers Telugu (Telugu script) in its 100-language pretraining set."
        elif lang_lower in ("malayalam", "ml", "mal"):
            return "XLM-R officially covers Malayalam (Malayalam script) in its 100-language pretraining set."
        elif lang_lower in ("kannada", "kn", "kan"):
            return "XLM-R officially covers Kannada (Kannada script) in its 100-language pretraining set."
        elif lang_lower in ("assamese", "as", "asm"):
            return "XLM-R officially covers Assamese (Bengali script) in its 100-language pretraining set."
        elif lang_lower in ("tulu", "tcy"):
            return (
                "XLM-R does NOT include Tulu in its pretraining language list. "
                "Kannada-script Tulu tokens rely on Kannada subword piece decomposition."
            )
        elif lang_lower in ("santali", "sat"):
            return (
                "XLM-R does NOT have explicit Ol Chiki pretraining. "
                "Characters will fall back to byte-level BPE representations."
            )
        elif lang_lower in ("bhojpuri", "bho", "dogri", "doi", "konkani", "kok", "bodo", "brx"):
            return (
                f"XLM-R does NOT isolate {language} as a dedicated pretraining corpus. "
                f"Relies on cross-lingual transfer via shared Devanagari script representations with Hindi/Marathi."
            )
        return f"Language '{language}' coverage in XLM-R requires verification against vocabulary."

    def _pool_embeddings(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pools token embeddings into a sentence representation."""
        if self.pooling_method == "cls":
            return last_hidden_state[:, 0, :]
        elif self.pooling_method == "max":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            last_hidden_state[input_mask_expanded == 0] = -1e9
            return torch.max(last_hidden_state, 1)[0]
        else:  # 'mean'
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

    def extract_embeddings(
        self,
        texts: List[str],
        sample_ids: Optional[List[str]] = None,
    ) -> TextEmbeddingOutput:
        """Extracts dense sentence embeddings across a batch of texts.

        Args:
            texts: List of preprocessed or raw strings.
            sample_ids: Optional list of sample identifiers.

        Returns:
            TextEmbeddingOutput: Numpy embeddings array and metadata.
        """
        self.load_encoder()
        sids = sample_ids or [f"idx_{i}" for i in range(len(texts))]

        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = [str(t or "") for t in texts[i : i + self.batch_size]]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                pooled = self._pool_embeddings(outputs.last_hidden_state, attention_mask)
                all_embeddings.append(pooled.cpu().numpy())

        embeddings_matrix = np.vstack(all_embeddings) if all_embeddings else np.empty((0, 768))

        return TextEmbeddingOutput(
            model_name=self.model_name,
            embeddings=embeddings_matrix,
            pooling_method=self.pooling_method,
            hidden_dimension=embeddings_matrix.shape[1] if embeddings_matrix.size > 0 else 768,
            num_samples=len(texts),
            sample_ids=sids,
        )

    def predict_sentiment(
        self,
        texts: List[str],
        sample_ids: Optional[List[str]] = None,
    ) -> List[TextClassificationOutput]:
        """Runs sentiment inference using sequence classification head.
        
        Reports explicitly if the model has not yet been fine-tuned on labeled regional data.
        """
        self.load_classifier()
        sids = sample_ids or [f"idx_{i}" for i in range(len(texts))]
        results: List[TextClassificationOutput] = []

        status_note = (
            "Model has been fine-tuned on labeled domain data."
            if self.is_classifier_trained
            else "Untrained / zero-shot classification head: predictions reflect pre-fine-tuning weights only. Fine-tuning required."
        )

        for i in range(0, len(texts), self.batch_size):
            batch_texts = [str(t or "") for t in texts[i : i + self.batch_size]]
            batch_ids = sids[i : i + self.batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.classifier_model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for sid, prob_vec in zip(batch_ids, probs):
                pred_idx = int(np.argmax(prob_vec))
                pred_label = ID2LABEL.get(pred_idx, "neutral")
                confidence = float(prob_vec[pred_idx])
                prob_dict = {ID2LABEL[j]: float(prob_vec[j]) for j in range(len(prob_vec))}

                results.append(
                    TextClassificationOutput(
                        sample_id=sid,
                        predicted_label=pred_label,
                        confidence=round(confidence, 4),
                        probabilities=prob_dict,
                        model_trained=self.is_classifier_trained,
                        status_note=status_note,
                    )
                )

        return results

    def fine_tune(
        self,
        train_records: List[TextRecord],
        val_records: Optional[List[TextRecord]] = None,
        epochs: int = 3,
        learning_rate: float = 2e-5,
    ) -> Dict[str, Any]:
        """Fine-tunes XLM-R classifier if labeled sentiment samples are present."""
        # Filter labeled samples
        labeled_train = [
            r for r in train_records
            if r.sentiment and r.sentiment != SentimentLabel.UNLABELED
        ]

        if len(labeled_train) < 6:
            logger.warning(
                f"Insufficient labeled training samples ({len(labeled_train)} found). "
                f"Minimum 6 required across classes. Reporting feature extraction mode only."
            )
            return {
                "trained": False,
                "reason": f"Insufficient labeled data ({len(labeled_train)} samples).",
                "status": "sentiment classification has not yet been trained",
            }

        self.load_classifier()
        self.classifier_model.train()
        optimizer = torch.optim.AdamW(self.classifier_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        texts = [r.processed_text or r.text for r in labeled_train]
        labels = [SENTIMENT_LABEL_MAP[r.sentiment] for r in labeled_train]

        logger.info(f"Fine-tuning XLM-RoBERTa on {len(texts)} samples for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(0, len(texts), self.batch_size):
                b_texts = texts[i : i + self.batch_size]
                b_labels = torch.tensor(labels[i : i + self.batch_size], dtype=torch.long).to(self.device)

                encoded = self.tokenizer(
                    b_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                optimizer.zero_grad()
                outputs = self.classifier_model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, b_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

        self.is_classifier_trained = True
        self.classifier_model.eval()
        return {
            "trained": True,
            "samples_count": len(texts),
            "epochs": epochs,
            "final_loss": round(total_loss, 4),
        }

    def save_checkpoint(self, output_dir: str) -> str:
        """Saves model weights and tokenizer configuration."""
        os.makedirs(output_dir, exist_ok=True)
        if self.classifier_model:
            self.classifier_model.save_pretrained(output_dir)
        elif self.encoder:
            self.encoder.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
        return output_dir

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Loads model weights and tokenizer from checkpoint directory."""
        logger.info(f"Loading checkpoint from {checkpoint_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        self.classifier_model.to(self.device)
        self.is_classifier_trained = True

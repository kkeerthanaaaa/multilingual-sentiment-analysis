"""Google MuRIL (Multilingual Representations for Indian Languages) pipeline.

Provides representation extraction and fine-grained 5-emotion classification
tailored for 17 Indian languages including Tamil, Telugu, Malayalam, Kannada,
Assamese, and dialectal transliterated/code-mixed regional text.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from src.data.schema import TextRecord, SentimentLabel
from src.text_processing.xlm_roberta import (
    TextEmbeddingOutput,
    TextClassificationOutput,
    ID2LABEL,
    LABEL2ID,
    SENTIMENT_LABEL_MAP,
)
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Official 17 Indian languages + English supported by Google MuRIL
MURIL_SUPPORTED_LANGUAGES = {
    "as", "bn", "gu", "hi", "kn", "ks", "ml", "mr", "ne",
    "or", "pa", "sa", "sd", "ta", "te", "ur", "en"
}


@dataclass
class MuRILLanguageCoverage:
    language: str
    is_supported: bool
    recommended_model: str
    script_notes: str


class MuRILProcessor:
    """MuRIL high-accuracy text processing and emotion classification pipeline."""

    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
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

    @classmethod
    def verify_language_coverage(cls, language: str) -> MuRILLanguageCoverage:
        """Verifies whether a language is in MuRIL's native 17-language pretraining set."""
        l = language.strip().lower()
        alias_map = {
            "tamil": "ta", "telugu": "te", "malayalam": "ml", "kannada": "kn",
            "assamese": "as", "bengali": "bn", "gujarati": "gu", "hindi": "hi",
            "marathi": "mr", "odia": "or", "punjabi": "pa", "urdu": "ur",
            "tulu": "tcy", "konkani": "kok", "santali": "sat", "bodo": "brx",
            "bhojpuri": "bho", "dogri": "doi"
        }
        code = alias_map.get(l, l)
        is_sup = code in MURIL_SUPPORTED_LANGUAGES

        if is_sup:
            rec = "google/muril-base-cased"
            notes = f"Officially supported in MuRIL with native script and Latin transliteration (code: {code})."
        elif code in ("tcy", "kok", "bho", "doi", "brx", "sat"):
            rec = "google/muril-base-cased"
            notes = f"Cross-lingual subword and script transfer via shared Indic script with {code}."
        else:
            rec = "Unknown"
            notes = f"Language '{language}' lacks direct verification."

        return MuRILLanguageCoverage(
            language=language,
            is_supported=is_sup,
            recommended_model=rec,
            script_notes=notes,
        )

    def load_encoder(self) -> None:
        """Loads MuRIL tokenizer and transformer encoder."""
        if self.tokenizer is None or self.encoder is None:
            logger.info(f"Loading MuRIL encoder '{self.model_name}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.encoder = AutoModel.from_pretrained(self.model_name)
            self.encoder.to(self.device)
            self.encoder.eval()

    def load_classifier(self, num_labels: int = 5) -> None:
        """Loads sequence classification head for MuRIL."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.classifier_model is None:
            logger.info(f"Loading MuRIL sequence classifier ({num_labels} labels) on {self.device}...")
            id2l = ID2LABEL if num_labels == 5 else {0: "negative", 1: "neutral", 2: "positive"}
            l2id = LABEL2ID if num_labels == 5 else {"negative": 0, "neutral": 1, "positive": 2}
            self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                id2label=id2l,
                label2id=l2id,
            )
            self.classifier_model.to(self.device)

    def _pool_embeddings(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pools token embeddings into sentence representations."""
        if self.pooling_method == "cls":
            return last_hidden_state[:, 0, :]
        elif self.pooling_method == "max":
            mask_exp = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            last_hidden_state[mask_exp == 0] = -1e9
            return torch.max(last_hidden_state, 1)[0]
        else:  # 'mean'
            mask_exp = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_emb = torch.sum(last_hidden_state * mask_exp, 1)
            sum_m = torch.clamp(mask_exp.sum(1), min=1e-9)
            return sum_emb / sum_m

    def extract_embeddings(
        self,
        texts: List[str],
        sample_ids: Optional[List[str]] = None,
    ) -> TextEmbeddingOutput:
        """Extracts dense sentence embeddings using MuRIL."""
        self.load_encoder()
        all_embeddings: List[np.ndarray] = []
        sids = sample_ids or [f"idx_{i}" for i in range(len(texts))]

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.encoder(**encoded)
                pooled = self._pool_embeddings(outputs.last_hidden_state, encoded["attention_mask"])
                all_embeddings.append(pooled.cpu().numpy())

        emb_matrix = np.vstack(all_embeddings) if all_embeddings else np.empty((0, 768))

        return TextEmbeddingOutput(
            model_name=self.model_name,
            embeddings=emb_matrix,
            pooling_method=self.pooling_method,
            hidden_dimension=emb_matrix.shape[1] if len(emb_matrix) > 0 else 0,
            num_samples=len(texts),
            sample_ids=sids,
        )

    def predict_sentiment(
        self,
        texts: List[str],
        sample_ids: Optional[List[str]] = None,
    ) -> List[TextClassificationOutput]:
        """Classifies text emotion/sentiment using fine-tuned MuRIL."""
        self.load_classifier(num_labels=5)
        self.classifier_model.eval()
        sids = sample_ids or [f"idx_{i}" for i in range(len(texts))]
        results: List[TextClassificationOutput] = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_ids = sids[i : i + self.batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = self.classifier_model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for sid, prob_vec in zip(batch_ids, probs):
                pred_idx = int(np.argmax(prob_vec))
                confidence = float(prob_vec[pred_idx])
                id2l = self.classifier_model.config.id2label
                pred_label = id2l.get(pred_idx, ID2LABEL.get(pred_idx, f"label_{pred_idx}"))
                prob_dict = {
                    id2l.get(j, ID2LABEL.get(j, f"label_{j}")): float(prob_vec[j])
                    for j in range(len(prob_vec))
                }

                note = (
                    "MuRIL supervised prediction."
                    if self.is_classifier_trained
                    else "Notice: Head untrained; fine-tuning required for authoritative prediction."
                )

                results.append(
                    TextClassificationOutput(
                        sample_id=sid,
                        predicted_label=pred_label,
                        confidence=round(confidence, 4),
                        probabilities=prob_dict,
                        model_trained=self.is_classifier_trained,
                        status_note=note,
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
        """Fine-tunes MuRIL sequence classification head on labeled Indic records."""
        labeled_train = [
            r for r in train_records
            if r.sentiment and r.sentiment != SentimentLabel.UNLABELED
        ]
        if len(labeled_train) < 6:
            raise ValueError(f"Insufficient labeled samples ({len(labeled_train)}). Minimum 6 required.")

        num_labels = 5
        self.load_classifier(num_labels=num_labels)
        self.classifier_model.train()

        optimizer = torch.optim.AdamW(self.classifier_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_texts = [r.processed_text or r.text for r in labeled_train]
        train_labels = [SENTIMENT_LABEL_MAP.get(r.sentiment.to_5_emotion(), 4) for r in labeled_train]

        history = []
        for ep in range(epochs):
            total_loss = 0.0
            steps = 0
            for i in range(0, len(train_texts), self.batch_size):
                b_texts = train_texts[i : i + self.batch_size]
                b_labels = torch.tensor(train_labels[i : i + self.batch_size], dtype=torch.long).to(self.device)

                encoded = self.tokenizer(
                    b_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                optimizer.zero_grad()
                logits = self.classifier_model(**encoded).logits
                loss = criterion(logits, b_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1

            avg_loss = total_loss / max(steps, 1)
            history.append({"epoch": ep + 1, "loss": round(avg_loss, 4)})
            logger.info(f"MuRIL Epoch {ep + 1}/{epochs} - Loss: {avg_loss:.4f}")

        self.is_classifier_trained = True
        return {"epochs_trained": epochs, "history": history, "model": self.model_name}

    def save_checkpoint(self, output_dir: Union[str, Path]) -> str:
        """Saves fine-tuned checkpoint."""
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        if self.classifier_model:
            self.classifier_model.save_pretrained(str(p))
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(p))
        return str(p)

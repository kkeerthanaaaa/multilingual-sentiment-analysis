"""IndicBERT comparative pipeline for Indian language representation."""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from src.data.schema import TextRecord, SentimentLabel
from src.text_processing.xlm_roberta import TextEmbeddingOutput, TextClassificationOutput, ID2LABEL, LABEL2ID, SENTIMENT_LABEL_MAP
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Factual language support tables for IndicBERT versions
INDICBERT_V1_LANGUAGES = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"}
INDICBERT_V2_LANGUAGES = {
    "as", "bn", "brx", "doi", "gom", "gu", "hi", "kn", "kas", "kok",
    "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur", "bho"
}


@dataclass
class IndicBERTLanguageCoverage:
    language: str
    v1_supported: bool
    v2_supported: bool
    recommended_model: str
    script_notes: str


class IndicBERTProcessor:
    """IndicBERT text processing pipeline for comparison with XLM-RoBERTa.
    
    Verifies language/script support before execution and keeps its configuration,
    tokenization, and classification weights decoupled from XLM-RoBERTa.
    """

    def __init__(
        self,
        model_name: str = "ai4bharat/IndicBERTv2-MLM-only",
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
    def verify_language_coverage(cls, language: str) -> IndicBERTLanguageCoverage:
        """Verifies whether a target language is covered by IndicBERT v1 or v2."""
        l = language.strip().lower()
        # Aliases mapping
        alias_map = {
            "assamese": "as", "tulu": "tcy", "konkani": "kok",
            "santali": "sat", "bodo": "brx", "bhojpuri": "bho", "dogri": "doi",
            "tamil": "ta", "telugu": "te", "malayalam": "ml", "kannada": "kn"
        }
        code = alias_map.get(l, l)

        v1_sup = code in INDICBERT_V1_LANGUAGES
        v2_sup = code in INDICBERT_V2_LANGUAGES

        if code == "tcy":
            rec = "None (Experimental Subword Transfer Only)"
            notes = "Tulu is NOT officially supported in IndicBERT v1 or v2. Kannada script tokens rely on Kannada vocabulary."
        elif v1_sup and v2_sup:
            rec = "ai4bharat/IndicBERTv2-MLM-only"
            notes = f"Fully supported in both IndicBERT v1 and IndicBERT v2 (code: {code})."
        elif v2_sup:
            rec = "ai4bharat/IndicBERTv2-MLM-only"
            notes = f"Fully included in IndicBERT v2 23-language pretraining set (code: {code})."
        elif v1_sup:
            rec = "ai4bharat/indic-bert"
            notes = f"Included in IndicBERT v1 (code: {code})."
        else:
            rec = "Unknown / Cross-lingual Transfer"
            notes = f"Language '{language}' lacks verified pretraining representation in IndicBERT."

        return IndicBERTLanguageCoverage(
            language=language,
            v1_supported=v1_sup,
            v2_supported=v2_sup,
            recommended_model=rec,
            script_notes=notes,
        )

    def load_encoder(self) -> None:
        """Loads IndicBERT tokenizer and encoder model lazily."""
        if self.tokenizer is None or self.encoder is None:
            logger.info(f"Loading IndicBERT encoder '{self.model_name}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.encoder = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            self.encoder.to(self.device)
            self.encoder.eval()

    def load_classifier(self, num_labels: int = 5) -> None:
        """Loads sequence classification head for IndicBERT."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.classifier_model is None:
            logger.info(f"Loading IndicBERT sequence classifier on {self.device}...")
            self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                trust_remote_code=True,
            )
            self.classifier_model.to(self.device)

    def _pool_embeddings(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pools token embeddings into a sentence representation."""
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
        """Extracts dense sentence embeddings using IndicBERT."""
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

        matrix = np.vstack(all_embeddings) if all_embeddings else np.empty((0, 768))

        return TextEmbeddingOutput(
            model_name=self.model_name,
            embeddings=matrix,
            pooling_method=self.pooling_method,
            hidden_dimension=matrix.shape[1] if matrix.size > 0 else 768,
            num_samples=len(texts),
            sample_ids=sids,
        )

    def predict_sentiment(
        self,
        texts: List[str],
        sample_ids: Optional[List[str]] = None,
    ) -> List[TextClassificationOutput]:
        """Runs sentiment inference using IndicBERT classification head."""
        self.load_classifier()
        sids = sample_ids or [f"idx_{i}" for i in range(len(texts))]
        results: List[TextClassificationOutput] = []

        status_note = (
            "Model fine-tuned on labeled domain data."
            if self.is_classifier_trained
            else "Untrained / zero-shot classification head on IndicBERT: fine-tuning required."
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
        """Fine-tunes IndicBERT classifier on labeled sentiment records."""
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

        logger.info(f"Fine-tuning IndicBERT on {len(texts)} samples for {epochs} epochs...")

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

            logger.info(f"IndicBERT Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

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
        logger.info(f"IndicBERT model saved to {output_dir}")
        return output_dir

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Loads model weights and tokenizer from checkpoint directory."""
        logger.info(f"Loading IndicBERT checkpoint from {checkpoint_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, trust_remote_code=True)
        self.classifier_model.to(self.device)
        self.is_classifier_trained = True

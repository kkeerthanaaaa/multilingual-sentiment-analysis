"""Sentiment model training pipeline for low-resource Indian languages.

Loads real-time multilingual datasets, applies linguistic preprocessing (preserving
negations, emojis, scripts, and regional slang), trains selected sentiment model,
evaluates per-language and overall accuracy/F1, and persists trained checkpoints.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.dataset_loader import DatasetLoader
from src.data.real_dataset_loader import RealDatasetFetcher
from src.data.schema import TextRecord, SentimentLabel
from src.preprocessing.text_preprocessing import TextPreprocessor
from src.text_processing.hybrid_classifier import HybridSentimentClassifier, TrainingMetrics
from src.text_processing.indicbert import IndicBERTProcessor
from src.text_processing.xlm_roberta import XLMRobertaProcessor
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def train_sentiment_model(
    dataset_path: Optional[str] = None,
    model_type: str = "hybrid",
    epochs: int = 3,
    test_size: float = 0.2,
    output_dir: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """End-to-end training pipeline on real-time multilingual sentiment datasets.

    Args:
        dataset_path: Path to dataset CSV/JSONL. If None, auto-fetches real benchmark data.
        model_type: 'hybrid' (recommended for low-resource CPU), 'indicbert', or 'xlm-roberta'.
        epochs: Training epochs for neural models.
        test_size: Fraction of samples reserved for test/validation evaluation.
        output_dir: Directory where trained model checkpoints and metrics will be saved.
        random_state: Random seed for reproducible dataset splits.

    Returns:
        Dict containing training summary, evaluation metrics, and saved model path.
    """
    logger.info("=" * 70)
    logger.info("STARTING MULTILINGUAL REGIONAL SENTIMENT MODEL TRAINING PIPELINE")
    logger.info(f"Target Model: {model_type.upper()}")
    logger.info("=" * 70)

    # 1. Resolve Dataset
    loader = DatasetLoader()
    csv_path = dataset_path
    if not csv_path or not Path(csv_path).exists():
        default_real_csv = Path("data/raw/text/real_time_multilingual_sentiment_dataset.csv")
        if default_real_csv.exists():
            csv_path = str(default_real_csv)
            logger.info(f"Using existing real-time dataset at {csv_path}")
        else:
            logger.info("Fetching real-time benchmark datasets from AI4Bharat IndicSentiment...")
            fetcher = RealDatasetFetcher()
            _, fetched_path = fetcher.build_real_time_dataset(max_samples_per_lang=200)
            csv_path = str(fetched_path)

    # 2. Load & Validate Records
    df, records, report = loader.load_text_dataset(csv_path)
    logger.info(f"Loaded {len(records)} records from {csv_path} (Clean: {report.is_clean if report else True})")

    # Filter labeled records
    labeled_records = [
        r for r in records
        if r.sentiment and r.sentiment != SentimentLabel.UNLABELED
    ]
    if len(labeled_records) < 6:
        raise ValueError(f"Insufficient labeled records for training ({len(labeled_records)} found).")

    # 3. Linguistic Preprocessing (Preserving negations, emojis, scripts)
    preprocessor = TextPreprocessor()
    for rec in labeled_records:
        if not rec.processed_text:
            prep_out = preprocessor.preprocess(rec.text, sample_id=rec.sample_id)
            rec.processed_text = prep_out.processed_text

    # 4. Train / Test Split Stratified where possible
    raw_texts = [r.processed_text or r.text for r in labeled_records]
    raw_labels = [
        r.sentiment.to_5_emotion().value.lower()
        if hasattr(r.sentiment, "to_5_emotion")
        else r.sentiment.value.lower()
        for r in labeled_records
    ]
    raw_langs = [r.language.lower() for r in labeled_records]

    # Combine label + lang for stratification if sample count permits
    strat_key = [f"{lbl}_{lng}" for lbl, lng in zip(raw_labels, raw_langs)]
    # Fallback to label-only stratification if singletons exist
    counts = pd.Series(strat_key).value_counts()
    if (counts < 2).any():
        strat_key = raw_labels

    train_indices, test_indices = train_test_split(
        range(len(labeled_records)),
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key,
    )

    train_recs = [labeled_records[i] for i in train_indices]
    test_recs = [labeled_records[i] for i in test_indices]

    train_texts = [train_recs[i].processed_text or train_recs[i].text for i in range(len(train_recs))]
    train_labels = [
        train_recs[i].sentiment.to_5_emotion().value.lower()
        if hasattr(train_recs[i].sentiment, "to_5_emotion")
        else train_recs[i].sentiment.value.lower()
        for i in range(len(train_recs))
    ]
    train_langs = [train_recs[i].language.lower() for i in range(len(train_recs))]

    test_texts = [test_recs[i].processed_text or test_recs[i].text for i in range(len(test_recs))]
    test_labels = [
        test_recs[i].sentiment.to_5_emotion().value.lower()
        if hasattr(test_recs[i].sentiment, "to_5_emotion")
        else test_recs[i].sentiment.value.lower()
        for i in range(len(test_recs))
    ]
    test_langs = [test_recs[i].language.lower() for i in range(len(test_recs))]

    logger.info(f"Dataset split: {len(train_texts)} training samples, {len(test_texts)} test samples.")
    logger.info(f"Target languages in split: {sorted(list(set(raw_langs)))}")

    # 5. Model Training & Evaluation
    m_type = model_type.strip().lower()
    default_out_dir = Path("models") / f"{m_type}_sentiment"
    save_dir = Path(output_dir) if output_dir else default_out_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    result_summary: Dict[str, Any] = {}

    if m_type in ("hybrid", "hybrid_classifier"):
        classifier = HybridSentimentClassifier()
        metrics: TrainingMetrics = classifier.fit(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=test_texts,
            val_labels=test_labels,
            val_languages=test_langs,
        )
        model_file = classifier.save(save_dir)

        result_summary = {
            "model_type": "hybrid",
            "model_path": str(model_file),
            "train_samples": metrics.train_samples,
            "test_samples": metrics.test_samples,
            "overall_accuracy": metrics.overall_accuracy,
            "macro_f1": metrics.macro_f1,
            "macro_precision": metrics.macro_precision,
            "macro_recall": metrics.macro_recall,
            "per_language_accuracy": metrics.per_language_accuracy,
            "classification_report": metrics.classification_report_str,
            "classes": metrics.classes,
        }

    elif m_type in ("indicbert", "indic-bert"):
        proc = IndicBERTProcessor()
        fit_res = proc.fine_tune(train_recs, val_records=test_recs, epochs=epochs)
        proc.save_checkpoint(str(save_dir))

        # Evaluate on test split
        preds = proc.predict_sentiment(test_texts)
        pred_labels = [p.predicted_label.lower() for p in preds]
        correct = sum(1 for p, y in zip(pred_labels, test_labels) if p == y)
        acc = round(correct / len(test_labels), 4) if test_labels else 0.0

        result_summary = {
            "model_type": "indicbert",
            "model_path": str(save_dir),
            "train_samples": len(train_recs),
            "test_samples": len(test_recs),
            "overall_accuracy": acc,
            "epochs": epochs,
            "fit_metrics": fit_res,
        }

    elif m_type in ("xlm_roberta", "xlm-roberta"):
        proc = XLMRobertaProcessor()
        fit_res = proc.fine_tune(train_recs, val_records=test_recs, epochs=epochs)
        proc.save_checkpoint(str(save_dir))

        preds = proc.predict_sentiment(test_texts)
        pred_labels = [p.predicted_label.lower() for p in preds]
        correct = sum(1 for p, y in zip(pred_labels, test_labels) if p == y)
        acc = round(correct / len(test_labels), 4) if test_labels else 0.0

        result_summary = {
            "model_type": "xlm-roberta",
            "model_path": str(save_dir),
            "train_samples": len(train_recs),
            "test_samples": len(test_recs),
            "overall_accuracy": acc,
            "epochs": epochs,
            "fit_metrics": fit_res,
        }
    elif m_type in ("muril", "google-muril", "google/muril-base-cased"):
        from src.text_processing.muril import MuRILProcessor
        proc = MuRILProcessor()
        fit_res = proc.fine_tune(train_recs, val_records=test_recs, epochs=epochs)
        proc.save_checkpoint(str(save_dir))

        preds = proc.predict_sentiment(test_texts)
        pred_labels = [p.predicted_label.lower() for p in preds]
        correct = sum(1 for p, y in zip(pred_labels, test_labels) if p == y)
        acc = round(correct / len(test_labels), 4) if test_labels else 0.0

        result_summary = {
            "model_type": "muril",
            "model_path": str(save_dir),
            "train_samples": len(train_recs),
            "test_samples": len(test_recs),
            "overall_accuracy": acc,
            "epochs": epochs,
            "fit_metrics": fit_res,
        }
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'hybrid', 'muril', 'indicbert', or 'xlm-roberta'.")

    # Save metrics report JSON
    report_file = save_dir / "evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info(f"MODEL TRAINING COMPLETE. Results saved to {save_dir}")
    logger.info(f"Overall Accuracy: {result_summary.get('overall_accuracy', 0):.2%}")
    if "macro_f1" in result_summary:
        logger.info(f"Macro F1 Score:   {result_summary['macro_f1']:.4f}")
    if "per_language_accuracy" in result_summary:
        logger.info(f"Per-Language Accuracy: {result_summary['per_language_accuracy']}")
    logger.info("=" * 70)

    return result_summary

"""Unit tests for XLM-RoBERTa and IndicBERT text processing pipelines."""

import pytest
import torch
import numpy as np
from src.text_processing.xlm_roberta import XLMRobertaProcessor, TextEmbeddingOutput
from src.text_processing.indicbert import IndicBERTProcessor
from src.data.schema import TextRecord, SentimentLabel


def test_xlm_roberta_language_coverage_reporting():
    """Verifies that XLM-R coverage notices factually state language representation realities."""
    tulu_notice = XLMRobertaProcessor.get_language_coverage_notice("tulu")
    assert "not include tulu" in tulu_notice.lower()
    assert "kannada subword" in tulu_notice.lower()

    santali_notice = XLMRobertaProcessor.get_language_coverage_notice("santali")
    assert "ol chiki" in santali_notice.lower()

    asm_notice = XLMRobertaProcessor.get_language_coverage_notice("assamese")
    assert "officially covers assamese" in asm_notice.lower()


def test_indicbert_language_coverage_verification():
    """Verifies IndicBERT language coverage verification matrix."""
    cov_asm = IndicBERTProcessor.verify_language_coverage("assamese")
    assert cov_asm.v1_supported is True
    assert cov_asm.v2_supported is True

    cov_tcy = IndicBERTProcessor.verify_language_coverage("tulu")
    assert cov_tcy.v1_supported is False
    assert cov_tcy.v2_supported is False

    cov_bho = IndicBERTProcessor.verify_language_coverage("bhojpuri")
    assert cov_bho.v2_supported is True


def test_embedding_pooling_methods():
    """Verifies that mean, cls, and max pooling operations compute valid tensor dimensions."""
    proc = XLMRobertaProcessor(pooling_method="mean")

    batch_size = 2
    seq_len = 5
    hidden_dim = 16

    # Fake hidden states and attention mask
    hidden_states = torch.ones((batch_size, seq_len, hidden_dim))
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

    # Mean pooling
    proc.pooling_method = "mean"
    mean_pooled = proc._pool_embeddings(hidden_states, attention_mask)
    assert mean_pooled.shape == (batch_size, hidden_dim)

    # CLS pooling
    proc.pooling_method = "cls"
    cls_pooled = proc._pool_embeddings(hidden_states, attention_mask)
    assert cls_pooled.shape == (batch_size, hidden_dim)

    # Max pooling
    proc.pooling_method = "max"
    max_pooled = proc._pool_embeddings(hidden_states, attention_mask)
    assert max_pooled.shape == (batch_size, hidden_dim)


def test_fine_tune_insufficient_labeled_data_guard():
    """Verifies that attempting to fine-tune without sufficient labeled data reports untrained status."""
    proc = XLMRobertaProcessor()
    # Provide only 2 samples (below minimum required 6)
    records = [
        TextRecord(sample_id="1", text="Good", language="tulu", sentiment=SentimentLabel.POSITIVE),
        TextRecord(sample_id="2", text="Bad", language="tulu", sentiment=SentimentLabel.NEGATIVE),
    ]

    res = proc.fine_tune(records)
    assert res["trained"] is False
    assert "sentiment classification has not yet been trained" in res["status"]

"""
Feature extraction for the detection v2 traditional-ML tier (STEP 3).

BASE features (length-blind, no raw token count anywhere):
  - TF-IDF word n-grams (1,2)
  - TF-IDF char n-grams (3,5)
  - type-token ratio (unique words / total words -- a ratio, not a count)
  - mean sentence length (words per sentence -- a ratio, not a raw count)
  - punctuation rates: count of each punctuation mark / total characters
    (a rate, not a raw count)

PLUS features = BASE + grammatical_proxy_final's 10 individual feature
COUNTS (raw counts, as specified -- not densities; the band-matched genres
sit in a narrow token-length range so residual length correlation here is
minor and disclosed, not corrected for) + marker_density (already a
density, from src.analysis.metrics).

Both feature sets exclude n_tokens/n_words entirely -- length diagnosis is
STEP 2's job, not folded into these models (hard rule).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.grammatical_proxy_final import grammatical_aave_proxy_final_breakdown
from src.analysis.metrics import marker_density

_SENT_SPLIT_RE = re.compile(r"[.!?]+")
_PUNCT_CHARS = ".,!?;:\"'-()"


def _type_token_ratio(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    return len(set(t.lower() for t in tokens)) / len(tokens)


def _mean_sentence_length(text: str) -> float:
    sents = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not sents:
        return float(len(text.split()))
    return len(text.split()) / len(sents)


def _punctuation_rates(text: str) -> dict:
    n_chars = max(len(text), 1)
    return {f"punct_rate_{ch}": text.count(ch) / n_chars for ch in _PUNCT_CHARS}


def base_numeric_features(texts: pd.Series) -> pd.DataFrame:
    rows = []
    for t in texts.astype(str):
        row = {"ttr": _type_token_ratio(t), "mean_sent_len": _mean_sentence_length(t)}
        row.update(_punctuation_rates(t))
        rows.append(row)
    return pd.DataFrame(rows)


def plus_numeric_features(texts: pd.Series) -> pd.DataFrame:
    base = base_numeric_features(texts)
    extra_rows = []
    for t in texts.astype(str):
        breakdown = grammatical_aave_proxy_final_breakdown(t)
        row = dict(breakdown)
        row["marker_density"] = marker_density(t)
        extra_rows.append(row)
    extra = pd.DataFrame(extra_rows)
    return pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)


class FeatureBuilder:
    """Fits TF-IDF (word+char) on a TRAIN text series, then transforms any
    text series into a combined sparse feature matrix (TF-IDF + numeric).
    feature_set: "BASE" or "PLUS"."""

    def __init__(self, feature_set: str, max_word_features: int = 3000, max_char_features: int = 3000):
        assert feature_set in ("BASE", "PLUS")
        self.feature_set = feature_set
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=max_word_features, sublinear_tf=True)
        self.char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=max_char_features, sublinear_tf=True)
        self._numeric_cols = None

    def _numeric(self, texts: pd.Series) -> pd.DataFrame:
        return plus_numeric_features(texts) if self.feature_set == "PLUS" else base_numeric_features(texts)

    def fit(self, texts: pd.Series):
        texts = texts.astype(str)
        self.word_vec.fit(texts)
        self.char_vec.fit(texts)
        numeric = self._numeric(texts)
        self._numeric_cols = list(numeric.columns)
        return self

    def transform(self, texts: pd.Series):
        texts = texts.astype(str)
        w = self.word_vec.transform(texts)
        c = self.char_vec.transform(texts)
        numeric = self._numeric(texts)[self._numeric_cols].fillna(0.0)
        n = sp.csr_matrix(numeric.values.astype(float))
        return sp.hstack([w, c, n], format="csr")

    def fit_transform(self, texts: pd.Series):
        self.fit(texts)
        return self.transform(texts)

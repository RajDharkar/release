"""
Corrected dialect-feature encoding for Experiment D, testing whether
Experiment A's near-zero PLUS-vs-BASE AUC delta was a property of the
features or an artifact of how they were encoded (see D1 audit in
reports/detection_v3_experiment_d.md for the full finding).

Two defects identified in src/detection/features.py's PLUS encoding:
  (a) the 10 grammatical_proxy_final features entered as RAW COUNTS, not
      length-normalized rates -- scale-with-document-length confound.
  (b) nothing was standardized: TF-IDF (~0-1 range post L2-norm) sat next
      to raw counts (~0-5 typical) and marker_density (~0.00-0.05) in an
      L2-regularized LogisticRegression with NO per-feature scaling, which
      penalizes small-magnitude features' coefficients out of proportion
      to their true informativeness.

This module fixes both: dialect_rate_features() returns 11 RATES (per 100
words, raw token count itself never appears as a feature -- only used as
the internal normalization denominator, consistent with the hard rule
against raw-count features). FeatureBuilderV2 optionally standardizes the
dialect block (fit on TRAIN only) and lets word/char TF-IDF be toggled
independently, for the D2 (features-only), D3 (three-way), and D4
(word-only redundancy) comparisons.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.grammatical_proxy_final import grammatical_aave_proxy_final_breakdown
from src.analysis.metrics import marker_density

DIALECT_RATE_COLS = [
    "habitual_be_90_rate", "zero_copula_progressive_174_rate", "aint_155_156_157_rate",
    "negative_concord_154_rate", "completive_done_104_rate", "regularized_reflexive_11_rate",
    "invariant_dont_158_rate", "regularized_past_128_rate", "double_modal_121_rate",
    "was_leveling_plural_180_rate", "marker_density_rate",
]


def dialect_rate_features(texts: pd.Series) -> pd.DataFrame:
    """11 dialect features, ALL as rates per 100 words (not raw counts).
    Word count is used only as the normalization denominator here -- it
    never appears itself as a model feature."""
    rows = []
    for t in texts.astype(str):
        n_words = max(len(t.split()), 1)
        breakdown = grammatical_aave_proxy_final_breakdown(t)
        row = {f"{k}_rate": (v / n_words) * 100 for k, v in breakdown.items()}
        row["marker_density_rate"] = marker_density(t) * 100  # already hits/word_count; x100 for the same "per 100 words" unit
        rows.append(row)
    return pd.DataFrame(rows)[DIALECT_RATE_COLS]


class FeatureBuilderV2:
    """Configurable builder: use_word / use_char TF-IDF, use_dialect (rates,
    optionally standardized). No raw token count in any configuration."""

    def __init__(self, use_word: bool, use_char: bool, use_dialect: bool,
                 scale_dialect: bool = True, max_word_features: int = 3000, max_char_features: int = 3000):
        self.use_word = use_word
        self.use_char = use_char
        self.use_dialect = use_dialect
        self.scale_dialect = scale_dialect
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=max_word_features, sublinear_tf=True) if use_word else None
        self.char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=max_char_features, sublinear_tf=True) if use_char else None
        self.scaler = StandardScaler() if (use_dialect and scale_dialect) else None

    def fit(self, texts: pd.Series):
        texts = texts.astype(str)
        if self.word_vec is not None:
            self.word_vec.fit(texts)
        if self.char_vec is not None:
            self.char_vec.fit(texts)
        if self.use_dialect and self.scaler is not None:
            self.scaler.fit(dialect_rate_features(texts).values)
        return self

    def transform(self, texts: pd.Series):
        texts = texts.astype(str)
        blocks = []
        if self.word_vec is not None:
            blocks.append(self.word_vec.transform(texts))
        if self.char_vec is not None:
            blocks.append(self.char_vec.transform(texts))
        if self.use_dialect:
            d = dialect_rate_features(texts).values.astype(float)
            if self.scaler is not None:
                d = self.scaler.transform(d)
            blocks.append(sp.csr_matrix(d))
        if not blocks:
            raise ValueError("no feature blocks enabled")
        if len(blocks) == 1:
            return blocks[0].tocsr()
        return sp.hstack(blocks, format="csr")

    def fit_transform(self, texts: pd.Series):
        self.fit(texts)
        return self.transform(texts)

    def n_dims(self) -> int:
        n = 0
        if self.word_vec is not None:
            n += len(self.word_vec.get_feature_names_out())
        if self.char_vec is not None:
            n += len(self.char_vec.get_feature_names_out())
        if self.use_dialect:
            n += len(DIALECT_RATE_COLS)
        return n

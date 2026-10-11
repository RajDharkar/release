"""Step 8: measured dialect-profile fidelity of the frozen synthetic arm.

This analysis performs feature extraction only. It reuses the frozen
orthographic and morphosyntactic measurement implementations unchanged and
does not invoke detectors, GPTZero, or Multi-VALUE transformation code.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.grammatical_proxy_final import (
    grammatical_aave_proxy_final,
    grammatical_aave_proxy_final_breakdown,
)
from src.loaders.source6_pd_poetry import MARKER_RE, _marker_density


ANALYSIS = ROOT / "data/analysis"
REPORT = ROOT / "reports/rq1_measured_dialect_profile_fidelity.md"
PROFILES_OUT = ANALYSIS / "rq1_measured_dialect_profiles.csv"
FEATURE_OUT = ANALYSIS / "rq1_measured_dialect_feature_summary.csv"
DISTANCE_OUT = ANALYSIS / "rq1_measured_dialect_profile_distances.csv"
FIGURE = ROOT / "figures/rq1_measured_dialect_profile_fidelity.pdf"

AAVE_PATH = ROOT / "data/processed/aave_poetry.csv"
SAE_PATH = ROOT / "data/processed/sae_poetry.csv"
CANONICAL_PATH = ANALYSIS / "rq1_proxy_sae521_multivalue_canonical.csv"
STEP1_SCORES_PATH = ANALYSIS / "rq1_full_poetry_natural_scores.csv"
PAIRED_SCORES_PATH = ANALYSIS / "rq1_proxy_original_canonical_paired_scores.csv"
BATCH_ORDER_PATH = ANALYSIS / "rq1_proxy_sae521_batch_order.csv"

EXPECTED_HASHES = {
    AAVE_PATH: "fbea79a240520823337b3d4a7898c79da3283f3d7d712317a192b0a9732badba",
    SAE_PATH: "d95e797a391a771c2d1b141c7ca71b01f4955751a18a2affeda3457ce4d0efee",
    CANONICAL_PATH: "d62929fa63d91cb6fc5e1007a9a5fb6a66490aa7deec31de48717d25cc5de73e",
    STEP1_SCORES_PATH: "6e835ee0280611c3b0d70f1ea7054a84e28ed636128068e2f2c11ecaebf3d208",
    PAIRED_SCORES_PATH: "e81d12cfdeccec509943a0937d1f8acd451865fcae123043b49a2cf3775cbb79",
    BATCH_ORDER_PATH: "220d50256499df2a963a5de5b3e84a7eedae5e21b14a5d4bab60b67b50910b37",
}

SEED = 20260919
N_BOOT = 10_000
BATCH_SIZE = 250
GROUPS = ["natural_AAVE", "original_SAE", "transformed_SAE"]
SOURCES = ["dunbar", "henderson", "hughes", "johnson1922"]
POOLED_WEIGHTS = {
    "dunbar": 432 / 733,
    "henderson": 51 / 733,
    "hughes": 229 / 733,
    "johnson1922": 21 / 733,
}
LARGE_WEIGHTS = {"dunbar": 432 / 661, "hughes": 229 / 661}

ORTH_FEATURES = [
    ("orth_de", "de", "de"),
    ("orth_dat", "dat", "dat"),
    ("orth_dis", "dis", "dis"),
    ("orth_dem", "dem", "dem"),
    ("orth_gwine", "gwine", "gwine"),
    ("orth_wid", "wid", "wid"),
    ("orth_yo_apostrophe", "yo'", "yo'"),
    ("orth_hyeah", "hyeah", "hyeah"),
    ("orth_ol_apostrophe", "ol'", "ol'"),
    ("orth_an_apostrophe", "an'", "an'"),
    ("orth_w_en", "w'en", "w'en"),
    ("orth_dey", "dey", "dey"),
    ("orth_sho_apostrophe", "sho'", "sho'"),
    ("orth_chile", "chile", "chile"),
    ("orth_mah", "mah", "mah"),
    ("orth_dar", "dar", "dar"),
    ("orth_daih", "daih", "daih"),
    ("orth_erbout", "erbout", "erbout"),
    ("orth_nuffin", "nuffin", "nuffin"),
    ("orth_fu_apostrophe", "fu'", "fu'"),
    ("orth_fuh", "fuh", "fuh"),
    ("orth_da", "da", "da"),
    ("orth_bekaze", "bekaze", "bekaze"),
    ("orth_knowed", "knowed", "knowed"),
    ("orth_natchel", "natchel", "natchel"),
]
ORTH_BY_MATCH = {matched: feature_id for feature_id, matched, _ in ORTH_FEATURES}

MORPH_DISPLAY = {
    "habitual_be_90": "Habitual be (eWAVE 90)",
    "zero_copula_progressive_174": "Zero copula, progressive (174)",
    "aint_155_156_157": "Ain't forms (155/156/157)",
    "negative_concord_154": "Negative concord (154)",
    "completive_done_104": "Completive done (104)",
    "regularized_reflexive_11": "Regularized reflexive (11)",
    "invariant_dont_158": "Invariant don't (158)",
    "regularized_past_128": "Regularized past (128)",
    "double_modal_121": "Double modal (121)",
    "was_leveling_plural_180": "Plural was leveling (180)",
}
MORPH_FEATURES = list(grammatical_aave_proxy_final_breakdown("").keys())
FEATURE_IDS = [x[0] for x in ORTH_FEATURES] + MORPH_FEATURES
RATE_COLUMNS = [f"rate_{x}" for x in FEATURE_IDS]

ZERO_SHOT_DETECTORS = [
    "Log Probability (GPT2-XL)", "Rank (negated) (GPT2-XL)",
    "Log-Rank (negated) (GPT2-XL)", "Entropy (GPT2-XL)",
    "DetectLLM-LRR (GPT2-XL)", "Binoculars (GPT2-XL)",
    "Log Probability (OPT-1.3B)", "Rank (negated) (OPT-1.3B)",
    "Log-Rank (negated) (OPT-1.3B)", "Entropy (OPT-1.3B)",
    "DetectLLM-LRR (OPT-1.3B)", "Binoculars (OPT-1.3B)",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sign_label(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path, expected in EXPECTED_HASHES.items():
        assert sha256_file(path) == expected, f"Frozen input hash mismatch: {path}"

    aave_all = pd.read_csv(AAVE_PATH, keep_default_na=False)
    aave = aave_all.loc[aave_all["variety"].eq("AAVE")].copy()
    sae = pd.read_csv(SAE_PATH, keep_default_na=False)
    canonical = pd.read_csv(CANONICAL_PATH, keep_default_na=False)
    scores = pd.read_csv(STEP1_SCORES_PATH)
    paired_scores = pd.read_csv(PAIRED_SCORES_PATH)
    order = pd.read_csv(BATCH_ORDER_PATH, keep_default_na=False)

    assert len(aave) == 212 and len(sae) == len(canonical) == len(order) == 521
    assert aave["id"].nunique() == 212 and sae["id"].nunique() == 521
    assert canonical["source_doc_id"].nunique() == 521
    assert canonical["transformation_success"].astype(bool).all()
    assert not canonical[["original_text", "transformed_text", "original_sha256",
                          "transformed_sha256", "executed_rule_count"]].isna().any().any()

    aave["source_doc_id"] = "poem_" + aave["id"].astype(str)
    sae["source_doc_id"] = "poem_sae_" + sae["id"].astype(str)
    aave["poem_id"] = aave["id"].astype(str)
    sae["poem_id"] = sae["id"].astype(str)

    frozen_meta = scores[["source_doc_id", "poem_id", "variety", "source_subset"]].drop_duplicates()
    assert len(frozen_meta) == 733 and frozen_meta["source_doc_id"].nunique() == 733
    for frame, variety in [(aave, "AAVE"), (sae, "SAE")]:
        check = frame[["source_doc_id", "poem_id", "source_subset"]].merge(
            frozen_meta.loc[frozen_meta["variety"].eq(variety)],
            on=["source_doc_id", "poem_id", "source_subset"], validate="one_to_one"
        )
        assert len(check) == len(frame)

    sae_ordered = sae.set_index("source_doc_id").loc[canonical["source_doc_id"]].reset_index()
    assert sae_ordered["poem_id"].astype(str).tolist() == canonical["poem_id"].astype(str).tolist()
    assert sae_ordered["text"].tolist() == canonical["original_text"].tolist()
    assert [sha256_text(x) for x in canonical["original_text"]] == canonical["original_sha256"].tolist()
    assert [sha256_text(x) for x in canonical["transformed_text"]] == canonical["transformed_sha256"].tolist()

    order_check = canonical[["canonical_batch_index", "source_doc_id", "poem_id", "sae_fold",
                             "original_sha256"]].merge(
        order.rename(columns={"source_text_sha256": "order_source_text_sha256"}),
        on=["canonical_batch_index", "source_doc_id", "poem_id", "sae_fold"],
        validate="one_to_one",
    )
    assert (order_check["original_sha256"] == order_check["order_source_text_sha256"]).all()

    paired_ids = paired_scores[["source_doc_id", "poem_id", "transformed_sha256"]].drop_duplicates()
    assert len(paired_ids) == 521 and not paired_ids["source_doc_id"].duplicated().any()
    pair_check = canonical[["source_doc_id", "poem_id", "transformed_sha256"]].merge(
        paired_ids, on=["source_doc_id", "poem_id", "transformed_sha256"], validate="one_to_one"
    )
    assert len(pair_check) == 521

    expected_sources = {
        "AAVE": {"dunbar": 115, "henderson": 47, "hughes": 38, "johnson1922": 12},
        "SAE": {"dunbar": 317, "henderson": 4, "hughes": 191, "johnson1922": 9},
    }
    assert aave["source_subset"].value_counts().to_dict() == expected_sources["AAVE"]
    assert sae["source_subset"].value_counts().to_dict() == expected_sources["SAE"]
    return aave, sae, canonical, paired_scores


def measure_text(text: str) -> dict[str, object]:
    text = str(text)
    word_count = len(text.split())
    assert word_count > 0
    marker_counts = Counter(match.group(0).casefold() for match in MARKER_RE.finditer(text))
    assert set(marker_counts).issubset(ORTH_BY_MATCH)
    morph_counts = grammatical_aave_proxy_final_breakdown(text)
    row: dict[str, object] = {
        "text_sha256": sha256_text(text),
        "word_count": word_count,
        "character_count": len(text),
        "orthographic_total_count": sum(marker_counts.values()),
        "orthographic_density": _marker_density(text),
        "morphosyntactic_total_count": sum(morph_counts.values()),
        "morphosyntactic_density": grammatical_aave_proxy_final(text),
    }
    for feature_id, matched, _ in ORTH_FEATURES:
        count = int(marker_counts[matched])
        row[f"count_{feature_id}"] = count
        row[f"rate_{feature_id}"] = count / word_count
    for feature_id in MORPH_FEATURES:
        count = int(morph_counts[feature_id])
        row[f"count_{feature_id}"] = count
        row[f"rate_{feature_id}"] = count / word_count
    assert row["orthographic_total_count"] == sum(row[f"count_{x[0]}"] for x in ORTH_FEATURES)
    assert row["morphosyntactic_total_count"] == sum(row[f"count_{x}"] for x in MORPH_FEATURES)
    assert np.isclose(row["orthographic_density"], row["orthographic_total_count"] / word_count,
                      rtol=0, atol=1e-15)
    assert np.isclose(row["morphosyntactic_density"], row["morphosyntactic_total_count"] / word_count,
                      rtol=0, atol=1e-15)
    return row


def build_profiles(aave: pd.DataFrame, sae: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    canonical_by_id = canonical.set_index("source_doc_id")
    for group, frame, text_col in [
        ("natural_AAVE", aave, "text"),
        ("original_SAE", sae, "text"),
    ]:
        for record in frame.itertuples(index=False):
            row = {
                "source_doc_id": record.source_doc_id,
                "poem_id": str(record.poem_id),
                "group": group,
                "source_subset": record.source_subset,
            }
            row.update(measure_text(getattr(record, text_col)))
            rows.append(row)
    sae_meta = sae.set_index("source_doc_id")
    for record in canonical.itertuples(index=False):
        row = {
            "source_doc_id": record.source_doc_id,
            "poem_id": str(record.poem_id),
            "group": "transformed_SAE",
            "source_subset": sae_meta.loc[record.source_doc_id, "source_subset"],
        }
        row.update(measure_text(record.transformed_text))
        rows.append(row)
    profiles = pd.DataFrame(rows)
    assert len(profiles) == 1254
    assert profiles.groupby("group").size().to_dict() == {
        "natural_AAVE": 212, "original_SAE": 521, "transformed_SAE": 521
    }
    assert not profiles.duplicated(["group", "source_doc_id"]).any()

    original = profiles.loc[profiles["group"].eq("original_SAE")].set_index("source_doc_id")
    transformed = profiles.loc[profiles["group"].eq("transformed_SAE")].set_index("source_doc_id")
    assert original.index.tolist() == sae["source_doc_id"].tolist()
    assert set(original.index) == set(transformed.index) == set(canonical_by_id.index)
    for density_col, cached_original, cached_transformed in [
        ("orthographic_density", "original_orthographic_density", "transformed_orthographic_density"),
        ("morphosyntactic_density", "original_morphosyntactic_density", "transformed_morphosyntactic_density"),
    ]:
        ids = canonical["source_doc_id"]
        assert np.allclose(original.loc[ids, density_col].to_numpy(float),
                           canonical[cached_original].to_numpy(float), rtol=0, atol=1e-15)
        assert np.allclose(transformed.loc[ids, density_col].to_numpy(float),
                           canonical[cached_transformed].to_numpy(float), rtol=0, atol=1e-15)

    paired_fields = {
        "paired_orthographic_density_change": transformed["orthographic_density"] - original["orthographic_density"],
        "paired_morphosyntactic_density_change": transformed["morphosyntactic_density"] - original["morphosyntactic_density"],
        "paired_word_count_change": transformed["word_count"] - original["word_count"],
        "paired_character_count_change": transformed["character_count"] - original["character_count"],
        "paired_executed_rule_count": canonical_by_id["executed_rule_count"].astype(int),
    }
    for column, values in paired_fields.items():
        profiles[column] = np.nan
        mapping = values.to_dict()
        paired_mask = profiles["group"].isin(["original_SAE", "transformed_SAE"])
        profiles.loc[paired_mask, column] = profiles.loc[paired_mask, "source_doc_id"].map(mapping)
    return profiles


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(values, [0.025, 0.975], method="linear")
    return float(lo), float(hi)


def paired_bootstrap(changes: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    boot_mean = np.empty(N_BOOT)
    boot_median = np.empty(N_BOOT)
    n = len(changes)
    for start in range(0, N_BOOT, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, N_BOOT)
        idx = rng.integers(0, n, size=(stop - start, n), dtype=np.int32)
        sampled = changes[idx]
        boot_mean[start:stop] = sampled.mean(axis=1)
        boot_median[start:stop] = np.median(sampled, axis=1)
    mean_lo, mean_hi = percentile_ci(boot_mean)
    median_lo, median_hi = percentile_ci(boot_median)
    return {
        "mean": float(changes.mean()), "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
        "median": float(np.median(changes)), "median_ci_lo": median_lo, "median_ci_hi": median_hi,
        "fraction_increasing": float((changes > 0).mean()),
        "fraction_decreasing": float((changes < 0).mean()),
        "fraction_unchanged": float((changes == 0).mean()),
    }


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    result = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    return 2.0 * float(result.statistic) / (len(x) * len(y)) - 1.0


def unpaired_bootstrap(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    boot_mean = np.empty(N_BOOT)
    boot_median = np.empty(N_BOOT)
    boot_rb = np.empty(N_BOOT)
    for start in range(0, N_BOOT, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, N_BOOT)
        bx = rng.integers(0, len(x), size=(stop - start, len(x)), dtype=np.int32)
        by = rng.integers(0, len(y), size=(stop - start, len(y)), dtype=np.int32)
        xs = x[bx]
        ys = y[by]
        boot_mean[start:stop] = xs.mean(axis=1) - ys.mean(axis=1)
        boot_median[start:stop] = np.median(xs, axis=1) - np.median(ys, axis=1)
        u = mannwhitneyu(xs, ys, alternative="two-sided", method="asymptotic", axis=1).statistic
        boot_rb[start:stop] = 2.0 * u / (len(x) * len(y)) - 1.0
    mean_lo, mean_hi = percentile_ci(boot_mean)
    median_lo, median_hi = percentile_ci(boot_median)
    rb_lo, rb_hi = percentile_ci(boot_rb)
    return {
        "mean_difference": float(x.mean() - y.mean()),
        "mean_difference_ci_lo": mean_lo, "mean_difference_ci_hi": mean_hi,
        "median_difference": float(np.median(x) - np.median(y)),
        "median_difference_ci_lo": median_lo, "median_difference_ci_hi": median_hi,
        "rank_biserial": rank_biserial(x, y),
        "rank_biserial_ci_lo": rb_lo, "rank_biserial_ci_hi": rb_hi,
    }


def group_distribution(values: np.ndarray) -> dict[str, float]:
    q25, q75 = np.quantile(values, [0.25, 0.75], method="linear")
    return {
        "n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
        "sd": float(values.std(ddof=1)), "q25": float(q25), "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def weighted_centroid(profiles: pd.DataFrame, group: str, weights: dict[str, float]) -> np.ndarray:
    subset = profiles.loc[profiles["group"].eq(group)]
    return sum(
        weight * subset.loc[subset["source_subset"].eq(source), RATE_COLUMNS].mean().to_numpy(float)
        for source, weight in weights.items()
    )


def distance_metrics(aave: np.ndarray, sae: np.ndarray, transformed: np.ndarray) -> dict[str, float]:
    target = aave - sae
    proxy_shift = transformed - sae
    d_original = float(np.abs(sae - aave).mean())
    d_transformed = float(np.abs(transformed - aave).mean())
    denom = float(np.linalg.norm(target) * np.linalg.norm(proxy_shift))
    cosine = float(np.dot(target, proxy_shift) / denom) if denom else np.nan
    return {
        "l1_d_original": d_original,
        "l1_d_transformed": d_transformed,
        "l1_distance_change": d_transformed - d_original,
        "l1_distance_ratio": d_transformed / d_original,
        "euclidean_d_original": float(np.linalg.norm(sae - aave)),
        "euclidean_d_transformed": float(np.linalg.norm(transformed - aave)),
        "cosine_target_proxy_shift": cosine,
    }


def profile_distance_bootstrap(profiles: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    a = profiles.loc[profiles["group"].eq("natural_AAVE"), RATE_COLUMNS].to_numpy(float)
    s = profiles.loc[profiles["group"].eq("original_SAE"), RATE_COLUMNS].to_numpy(float)
    t = profiles.loc[profiles["group"].eq("transformed_SAE")].set_index("source_doc_id")
    s_ids = profiles.loc[profiles["group"].eq("original_SAE"), "source_doc_id"].tolist()
    t = t.loc[s_ids, RATE_COLUMNS].to_numpy(float)
    d_orig = np.empty(N_BOOT)
    d_trans = np.empty(N_BOOT)
    for start in range(0, N_BOOT, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, N_BOOT)
        ai = rng.integers(0, len(a), size=(stop - start, len(a)), dtype=np.int32)
        pi = rng.integers(0, len(s), size=(stop - start, len(s)), dtype=np.int32)
        ac = a[ai].mean(axis=1)
        sc = s[pi].mean(axis=1)
        tc = t[pi].mean(axis=1)
        d_orig[start:stop] = np.abs(sc - ac).mean(axis=1)
        d_trans[start:stop] = np.abs(tc - ac).mean(axis=1)
    change = d_trans - d_orig
    ratio = d_trans / d_orig
    original_lo, original_hi = percentile_ci(d_orig)
    transformed_lo, transformed_hi = percentile_ci(d_trans)
    change_lo, change_hi = percentile_ci(change)
    ratio_lo, ratio_hi = percentile_ci(ratio)
    return {
        "l1_d_original_ci_lo": original_lo, "l1_d_original_ci_hi": original_hi,
        "l1_d_transformed_ci_lo": transformed_lo, "l1_d_transformed_ci_hi": transformed_hi,
        "l1_distance_change_ci_lo": change_lo, "l1_distance_change_ci_hi": change_hi,
        "l1_distance_ratio_ci_lo": ratio_lo, "l1_distance_ratio_ci_hi": ratio_hi,
        "bootstrap_proportion_distance_change_lt_zero": float((change < 0).mean()),
    }


def feature_summary(profiles: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    centroids: dict[str, dict[str, np.ndarray]] = {}
    centroids["unstandardized"] = {
        group: profiles.loc[profiles["group"].eq(group), RATE_COLUMNS].mean().to_numpy(float)
        for group in GROUPS
    }
    centroids["source_standardized_all4"] = {
        group: weighted_centroid(profiles, group, POOLED_WEIGHTS) for group in GROUPS
    }
    centroids["source_standardized_dunbar_hughes"] = {
        group: weighted_centroid(profiles, group, LARGE_WEIGHTS) for group in GROUPS
    }

    metadata = [
        (feature_id, "orthographic", display)
        for feature_id, _, display in ORTH_FEATURES
    ] + [
        (feature_id, "morphosyntactic", MORPH_DISPLAY[feature_id]) for feature_id in MORPH_FEATURES
    ]
    rows: list[dict[str, object]] = []
    for index, (feature_id, family, display) in enumerate(metadata):
        raw = centroids["unstandardized"]
        a = raw["natural_AAVE"][index]
        s = raw["original_SAE"][index]
        t = raw["transformed_SAE"][index]
        target = a - s
        shift = t - s
        if target == 0:
            direction = "NO_NUMERIC_TARGET"
        elif shift == 0:
            direction = "UNCHANGED"
        elif np.sign(target) == np.sign(shift):
            direction = "NATURAL_AAVE_DIRECTION"
        else:
            direction = "OPPOSITE_DIRECTION"
        original_gap = abs(a - s)
        transformed_gap = abs(a - t)
        if transformed_gap < original_gap:
            gap_movement = "TOWARD"
        elif transformed_gap > original_gap:
            gap_movement = "AWAY"
        else:
            gap_movement = "UNCHANGED"
        row: dict[str, object] = {
            "feature_id": feature_id, "feature_family": family, "feature_name": display,
            "natural_AAVE_mean_rate": a, "original_SAE_mean_rate": s,
            "transformed_SAE_mean_rate": t, "SAE_to_transformed_change": shift,
            "absolute_natural_minus_SAE_gap": original_gap,
            "absolute_natural_minus_transformed_gap": transformed_gap,
            "gap_movement": gap_movement,
            "target_direction": sign_label(target),
            "transformation_direction": sign_label(shift),
            "direction_fidelity": direction,
        }
        for spec in ["source_standardized_all4", "source_standardized_dunbar_hughes"]:
            label = "source_std" if spec == "source_standardized_all4" else "dunbar_hughes"
            row[f"{label}_natural_AAVE_mean_rate"] = centroids[spec]["natural_AAVE"][index]
            row[f"{label}_original_SAE_mean_rate"] = centroids[spec]["original_SAE"][index]
            row[f"{label}_transformed_SAE_mean_rate"] = centroids[spec]["transformed_SAE"][index]
        rows.append(row)
    return pd.DataFrame(rows), centroids


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan)
    valid = np.isfinite(values)
    p = values[valid]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    out[valid] = restored
    return out


def detector_diagnostics(profiles: pd.DataFrame, paired_scores: pd.DataFrame) -> pd.DataFrame:
    paired = profiles.loc[profiles["group"].eq("transformed_SAE"), [
        "source_doc_id", "paired_orthographic_density_change",
        "paired_morphosyntactic_density_change", "paired_executed_rule_count",
        "paired_word_count_change",
    ]]
    variables = [
        "paired_orthographic_density_change", "paired_morphosyntactic_density_change",
        "paired_executed_rule_count", "paired_word_count_change",
    ]
    rows: list[dict[str, object]] = []
    for detector in ZERO_SHOT_DETECTORS:
        d = paired_scores.loc[paired_scores["detector"].eq(detector), [
            "source_doc_id", "paired_change"
        ]].merge(paired, on="source_doc_id", validate="one_to_one")
        assert len(d) == 521
        for variable in variables:
            result = spearmanr(d["paired_change"], d[variable])
            rows.append({
                "detector": detector, "diagnostic_variable": variable, "n": len(d),
                "spearman_rho": float(result.statistic), "spearman_p": float(result.pvalue),
            })
    out = pd.DataFrame(rows)
    out["spearman_p_bh_48_tests"] = bh_adjust(out["spearman_p"].to_numpy(float))
    return out


def analyze(profiles: pd.DataFrame, paired_scores: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame, dict[str, object]
]:
    rng = np.random.default_rng(SEED)
    records: list[dict[str, object]] = []
    results: dict[str, object] = {"group_summaries": {}, "paired": {}, "comparisons": {}}

    for metric in ["orthographic_density", "morphosyntactic_density"]:
        results["group_summaries"][metric] = {}
        for group in GROUPS:
            values = profiles.loc[profiles["group"].eq(group), metric].to_numpy(float)
            summary = group_distribution(values)
            results["group_summaries"][metric][group] = summary
            records.append({"record_type": "group_distribution", "metric": metric,
                            "comparison": group, **summary})

        changes = profiles.loc[profiles["group"].eq("transformed_SAE"),
                               f"paired_{metric}_change"].to_numpy(float)
        paired = paired_bootstrap(changes, rng)
        results["paired"][metric] = paired
        records.append({"record_type": "paired_change", "metric": metric,
                        "comparison": "transformed_SAE_minus_original_SAE", "n": len(changes),
                        **paired, "bootstrap_replicates": N_BOOT, "bootstrap_seed": SEED})

        a = profiles.loc[profiles["group"].eq("natural_AAVE"), metric].to_numpy(float)
        for comparator in ["original_SAE", "transformed_SAE"]:
            c = profiles.loc[profiles["group"].eq(comparator), metric].to_numpy(float)
            comparison = unpaired_bootstrap(a, c, rng)
            results["comparisons"][(metric, comparator)] = comparison
            records.append({"record_type": "unpaired_comparison", "metric": metric,
                            "comparison": f"natural_AAVE_minus_{comparator}",
                            "n_natural_AAVE": len(a), "n_comparator": len(c), **comparison,
                            "bootstrap_replicates": N_BOOT, "bootstrap_seed": SEED})

    features, centroids = feature_summary(profiles)
    distance_rows: list[dict[str, object]] = []
    for spec, values in centroids.items():
        metrics = distance_metrics(values["natural_AAVE"], values["original_SAE"],
                                   values["transformed_SAE"])
        row = {"record_type": "profile_distance", "metric": "individual_feature_rate_centroids",
               "comparison": spec, "feature_count": len(FEATURE_IDS), **metrics}
        if spec == "unstandardized":
            boot = profile_distance_bootstrap(profiles, rng)
            row.update(boot)
            row.update({"bootstrap_replicates": N_BOOT, "bootstrap_seed": SEED})
        distance_rows.append(row)
    records.extend(distance_rows)

    diagnostic = detector_diagnostics(profiles, paired_scores)
    for row in diagnostic.to_dict("records"):
        records.append({"record_type": "detector_change_correlation",
                        "metric": "oriented_detector_score_change", "comparison": "paired_SAE",
                        **row})
    results["features"] = features
    results["centroids"] = centroids
    results["distances"] = {row["comparison"]: row for row in distance_rows}
    results["diagnostic"] = diagnostic
    return features, pd.DataFrame(records), results


def fmt(value: float, digits: int = 6, signed: bool = False) -> str:
    return f"{value:{'+' if signed else ''}.{digits}f}"


def make_figure(features: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    colors = {"natural_AAVE": "#0072B2", "original_SAE": "#555555", "transformed_SAE": "#D55E00"}
    markers = {"natural_AAVE": "^", "original_SAE": "s", "transformed_SAE": "o"}
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 10.5), gridspec_kw={"width_ratios": [1.35, 1.0]})
    panels = [
        (axes[0], features.loc[features["feature_family"].eq("orthographic")], "Orthographic markers"),
        (axes[1], features.loc[features["feature_family"].eq("morphosyntactic")], "Morphosyntactic features"),
    ]
    for ax, data, title in panels:
        data = data.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(data))
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.6, zorder=-2)
        for i, row in data.iterrows():
            style = "-" if row.gap_movement == "TOWARD" else ":" if row.gap_movement == "AWAY" else "--"
            ax.plot([row.original_SAE_mean_rate, row.transformed_SAE_mean_rate],
                    [i - 0.13, i], color="#777777", linestyle=style, linewidth=0.9, zorder=1)
        offsets = {"original_SAE": -0.13, "transformed_SAE": 0.0, "natural_AAVE": 0.13}
        for group, column in [
            ("original_SAE", "original_SAE_mean_rate"),
            ("transformed_SAE", "transformed_SAE_mean_rate"),
            ("natural_AAVE", "natural_AAVE_mean_rate"),
        ]:
            ax.scatter(data[column], y + offsets[group], marker=markers[group], s=31,
                       facecolor=colors[group], edgecolor="black", linewidth=0.35, zorder=3)
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for i, status in enumerate(data["gap_movement"]):
            ax.text(1.01, i, status.lower(), transform=trans, va="center", fontsize=6.8,
                    color="#333333", clip_on=False)
        ax.set_yticks(y, data["feature_name"], fontsize=7.3)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean feature hits per whitespace-delimited word", fontsize=8.5)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.06)
    legend = [
        Line2D([0], [0], marker=markers[group], color="none", markerfacecolor=colors[group],
               markeredgecolor="black", markersize=6, label=label)
        for group, label in [
            ("original_SAE", "Natural SAE"), ("transformed_SAE", "Transformed SAE"),
            ("natural_AAVE", "Natural AAVE")
        ]
    ] + [
        Line2D([0], [0], color="#777777", linestyle="-", label="SAE-to-transformed: toward"),
        Line2D([0], [0], color="#777777", linestyle=":", label="SAE-to-transformed: away"),
        Line2D([0], [0], color="#777777", linestyle="--", label="SAE-to-transformed: unchanged"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=8, frameon=True,
               bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("Measured dialect-feature profiles in natural and transformed poetry",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.047,
             "Status compares absolute distance from the natural AAVE feature mean; marker shape also identifies group.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.19, right=0.91, top=0.935, bottom=0.09, wspace=0.52)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def write_report(results: dict[str, object]) -> None:
    summaries = results["group_summaries"]
    paired = results["paired"]
    comparisons = results["comparisons"]
    features: pd.DataFrame = results["features"]
    distances = results["distances"]
    diagnostic: pd.DataFrame = results["diagnostic"]
    raw = distances["unstandardized"]
    source_std = distances["source_standardized_all4"]
    dh = distances["source_standardized_dunbar_hughes"]

    eligible = features.loc[features["direction_fidelity"].ne("NO_NUMERIC_TARGET")]
    direction_counts = eligible["direction_fidelity"].value_counts()
    agree = int(direction_counts.get("NATURAL_AAVE_DIRECTION", 0))
    opposite = int(direction_counts.get("OPPOSITE_DIRECTION", 0))
    unchanged = int(direction_counts.get("UNCHANGED", 0))
    agreement = agree / len(eligible)

    gap_results: dict[str, tuple[str, bool]] = {}
    for metric in ["orthographic_density", "morphosyntactic_density"]:
        original = comparisons[(metric, "original_SAE")]
        transformed = comparisons[(metric, "transformed_SAE")]
        def gap_relation(new_gap: float, old_gap: float) -> str:
            if np.isclose(new_gap, old_gap, rtol=0, atol=1e-15):
                return "unchanged"
            return "smaller" if new_gap < old_gap else "larger"

        mean_relation = gap_relation(abs(transformed["mean_difference"]), abs(original["mean_difference"]))
        median_relation = gap_relation(abs(transformed["median_difference"]), abs(original["median_difference"]))
        sign_changed = sign_label(original["mean_difference"]) != sign_label(transformed["mean_difference"])
        gap_results[metric] = (f"mean gap {mean_relation}; median gap {median_relation}", sign_changed)

    detector_pivot = diagnostic.pivot(index="detector", columns="diagnostic_variable", values="spearman_rho")
    profile_closer = raw["l1_distance_change"] < 0
    final_label = (
        "PROCEED TO INTEGRATED PROVENANCE SYNTHESIS"
        if profile_closer and raw["bootstrap_proportion_distance_change_lt_zero"] >= 0.95
        else "MEASURED PROFILE DOES NOT SUPPORT A PROXY-VALIDITY INTERPRETATION"
    )

    lines: list[str] = [
        "# RQ1 measured dialect-profile fidelity (Step 8)\n",
        "This frozen-text feature-extraction analysis asks whether canonical rule transformation moves the measured orthographic and morphosyntactic profile of SAE poetry toward the natural AAVE poetry population. It does not evaluate authenticity. Detector results did not select the features, weights, or distance metrics. No detector inference, GPTZero call, text regeneration, or manuscript edit occurred.\n",
        "## A. Input validation\n",
        "Validation passed for 212 natural AAVE poems, 521 natural SAE poems, and 521 canonical transformed versions paired one-to-one with those SAE poems. Source IDs, source subsets, original texts, per-text hashes, transformed hashes, and frozen file hashes agree with Steps 1-3 and 5. Historical Arm C texts are absent.\n",
        "Frozen input SHA-256 values:\n",
    ]
    for path, digest in EXPECTED_HASHES.items():
        lines.append(f"- `{path.relative_to(ROOT)}`: `{digest}`")

    lines += [
        "\n## B. Measurement definitions\n",
        "The exact existing implementations are reused unchanged: `src/loaders/source6_pd_poetry.py::_marker_density` and `src/analysis/grammatical_proxy_final.py::grammatical_aave_proxy_final`. Both densities equal matched hits divided by `len(text.split())`, so the denominator is whitespace-delimited words. No token normalization, lemmatization, stemming, or punctuation stripping occurs. Empty texts would score zero, but none are present. Case-insensitive regular expressions handle case; punctuation remains in the text and affects regex word-boundary behavior. Character count is Python `len(text)`. Neither measure uses a parser or language model.\n",
        "The orthographic density is the total count from one fixed case-insensitive regex over 25 identifiable surface forms. Step 8 retains counts and rates for: " + ", ".join(f"`{x[2]}`" for x in ORTH_FEATURES) + ". Individual counts sum exactly to the aggregate orthographic hit count.\n",
        "The finalized morphosyntactic proxy retains these 10 regex-based measured features:\n",
    ]
    for feature in MORPH_FEATURES:
        lines.append(f"- `{feature}`: {MORPH_DISPLAY[feature]}")
    lines.append("\nEach individual-feature rate is its document count divided by the same whitespace-word denominator. The aggregate morphosyntactic density is the sum of the 10 counts divided by that denominator. These are surface-pattern measurements with known false-positive and false-negative limits, not an authenticity scale.\n")

    lines += [
        "## C. Aggregate density distributions\n",
        "Means describe these measured densities only; they are not estimates of dialect authenticity.\n",
        "| density | group | n | mean | median | SD | Q25 | Q75 | IQR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in ["orthographic_density", "morphosyntactic_density"]:
        for group in GROUPS:
            row = summaries[metric][group]
            lines.append(
                f"| {metric} | {group} | {row['n']} | {row['mean']:.6f} | {row['median']:.6f} | "
                f"{row['sd']:.6f} | {row['q25']:.6f} | {row['q75']:.6f} | {row['iqr']:.6f} |"
            )

    lines += [
        "\n## D. Paired SAE-to-transformed changes\n",
        f"Paired percentile intervals use {N_BOOT:,} source-level replicates with seed `{SEED}`; each original/transformed pair is resampled together.\n",
        "| density | mean change [95% CI] | median change [95% CI] | increasing | decreasing | unchanged |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in ["orthographic_density", "morphosyntactic_density"]:
        row = paired[metric]
        lines.append(
            f"| {metric} | {row['mean']:+.6f} [{row['mean_ci_lo']:+.6f}, {row['mean_ci_hi']:+.6f}] | "
            f"{row['median']:+.6f} [{row['median_ci_lo']:+.6f}, {row['median_ci_hi']:+.6f}] | "
            f"{row['fraction_increasing']:.3f} | {row['fraction_decreasing']:.3f} | "
            f"{row['fraction_unchanged']:.3f} |"
        )

    lines += [
        "\n## E. Natural AAVE comparisons\n",
        "Natural AAVE comparisons are unpaired and descriptive, not causal dialect effects. Positive differences and rank-biserial values mean higher measured density in natural AAVE.\n",
        "| density | comparator | mean difference [95% CI] | median difference [95% CI] | rank-biserial [95% CI] | gap result |",
        "|---|---|---:|---:|---:|---|",
    ]
    for metric in ["orthographic_density", "morphosyntactic_density"]:
        for comparator in ["original_SAE", "transformed_SAE"]:
            row = comparisons[(metric, comparator)]
            gap_note = "baseline" if comparator == "original_SAE" else (
                gap_results[metric][0] + ("; mean sign changes" if gap_results[metric][1] else "; mean sign retained")
            )
            lines.append(
                f"| {metric} | {comparator} | {row['mean_difference']:+.6f} "
                f"[{row['mean_difference_ci_lo']:+.6f}, {row['mean_difference_ci_hi']:+.6f}] | "
                f"{row['median_difference']:+.6f} [{row['median_difference_ci_lo']:+.6f}, "
                f"{row['median_difference_ci_hi']:+.6f}] | {row['rank_biserial']:+.4f} "
                f"[{row['rank_biserial_ci_lo']:+.4f}, {row['rank_biserial_ci_hi']:+.4f}] | {gap_note} |"
            )

    lines += [
        "\n## F. Individual-feature profile\n",
        "Rates are hits per whitespace-delimited word. `Toward/away` compares absolute distance from the natural AAVE feature mean; it is distinct from direction agreement.\n",
        "| feature | family | natural AAVE | original SAE | transformed SAE | SAE-to-transformed | natural-SAE gap | natural-transformed gap | movement |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in features.itertuples(index=False):
        lines.append(
            f"| {row.feature_name} | {row.feature_family} | {row.natural_AAVE_mean_rate:.6f} | "
            f"{row.original_SAE_mean_rate:.6f} | {row.transformed_SAE_mean_rate:.6f} | "
            f"{row.SAE_to_transformed_change:+.6f} | {row.absolute_natural_minus_SAE_gap:.6f} | "
            f"{row.absolute_natural_minus_transformed_gap:.6f} | {row.gap_movement} |"
        )

    lines += [
        "\n## G. Feature-direction fidelity\n",
        f"Among {len(eligible)} features with a nonzero natural-AAVE-minus-SAE target direction, "
        f"**{agree}** move in the natural-AAVE direction, **{opposite}** move oppositely, and "
        f"**{unchanged}** are unchanged. The descriptive sign-agreement proportion is "
        f"**{agreement:.3f}** ({agree}/{len(eligible)}). Features are not treated as statistically independent.\n",
        "| direction result | count |",
        "|---|---:|",
        f"| natural-AAVE direction | {agree} |",
        f"| opposite direction | {opposite} |",
        f"| unchanged | {unchanged} |",
        f"| no numeric target difference | {int((features.direction_fidelity == 'NO_NUMERIC_TARGET').sum())} |",
    ]

    lines += [
        "\n## H. Multivariate measured-profile distance\n",
        f"The predeclared primary distance is the mean absolute difference across all {len(FEATURE_IDS)} frozen individual feature-rate centroids. The {N_BOOT:,}-replicate bootstrap independently resamples 212 AAVE poems and 521 SAE source IDs, keeping each original/transformed pair together (seed `{SEED}`). This is measured dialect-profile distance, not unrestricted linguistic distance.\n",
        "| quantity | point estimate | 95% bootstrap CI |",
        "|---|---:|---:|",
        f"| L1 mean distance: original SAE to natural AAVE | {raw['l1_d_original']:.8f} | [{raw['l1_d_original_ci_lo']:.8f}, {raw['l1_d_original_ci_hi']:.8f}] |",
        f"| L1 mean distance: transformed SAE to natural AAVE | {raw['l1_d_transformed']:.8f} | [{raw['l1_d_transformed_ci_lo']:.8f}, {raw['l1_d_transformed_ci_hi']:.8f}] |",
        f"| distance change (transformed minus original) | {raw['l1_distance_change']:+.8f} | [{raw['l1_distance_change_ci_lo']:+.8f}, {raw['l1_distance_change_ci_hi']:+.8f}] |",
        f"| distance ratio | {raw['l1_distance_ratio']:.6f} | [{raw['l1_distance_ratio_ci_lo']:.6f}, {raw['l1_distance_ratio_ci_hi']:.6f}] |",
        f"| proportion of replicates with distance change < 0 | {raw['bootstrap_proportion_distance_change_lt_zero']:.4f} | descriptive bootstrap proportion |",
        "\nSecondary point diagnostics:\n",
        "| diagnostic | original SAE | transformed SAE / shift |",
        "|---|---:|---:|",
        f"| Euclidean centroid distance to natural AAVE | {raw['euclidean_d_original']:.8f} | {raw['euclidean_d_transformed']:.8f} |",
        f"| cosine similarity: SAE-to-AAVE target vs SAE-to-transformed shift | n/a | {raw['cosine_target_proxy_shift']:+.4f} |",
        "\nThe cosine is near +1 only for closely aligned measured-feature movement, near 0 for an approximately orthogonal shift, and near -1 for an opposite shift. Sparse features limit interpretation. The bootstrap proportion is not a probability that the proxy is authentic.\n",
    ]

    lines += [
        "## I. Source-standardized sensitivity\n",
        "Each group centroid first averages rates within Dunbar, Henderson, Hughes, and Johnson, then combines those source means using the frozen pooled-corpus weights 432/733, 51/733, 229/733, and 21/733. This addresses coarse source-mixture imbalance, not all author, style, or topic differences.\n",
        "| specification | original L1 | transformed L1 | ratio | Euclidean original | Euclidean transformed | cosine |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| all-four-source standardized | {source_std['l1_d_original']:.8f} | {source_std['l1_d_transformed']:.8f} | "
        f"{source_std['l1_distance_ratio']:.6f} | {source_std['euclidean_d_original']:.8f} | "
        f"{source_std['euclidean_d_transformed']:.8f} | {source_std['cosine_target_proxy_shift']:+.4f} |",
    ]

    raw_direction = "toward" if raw["l1_distance_ratio"] < 1 else "away"
    dh_direction = "toward" if dh["l1_distance_ratio"] < 1 else "away"
    lines += [
        "\n## J. Dunbar/Hughes sensitivity\n",
        "The two-large-stratum analysis uses frozen renormalized weights 432/661 for Dunbar and 229/661 for Hughes.\n",
        "| specification | original L1 | transformed L1 | ratio | Euclidean original | Euclidean transformed | cosine |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Dunbar/Hughes standardized | {dh['l1_d_original']:.8f} | {dh['l1_d_transformed']:.8f} | "
        f"{dh['l1_distance_ratio']:.6f} | {dh['euclidean_d_original']:.8f} | "
        f"{dh['euclidean_d_transformed']:.8f} | {dh['cosine_target_proxy_shift']:+.4f} |",
        f"\nThe primary unstandardized L1 result moves {raw_direction}; the Dunbar/Hughes result moves {dh_direction}. "
        f"The coarse-source sensitivity {'changes' if raw_direction != dh_direction else 'does not change'} the direction of that point conclusion.\n",
    ]

    short_names = {
        "paired_orthographic_density_change": "orthographic density change",
        "paired_morphosyntactic_density_change": "morphosyntactic density change",
        "paired_executed_rule_count": "executed rules",
        "paired_word_count_change": "word-count change",
    }
    lines += [
        "## K. Detector-change diagnostic\n",
        "Spearman coefficients relate frozen Step 5 oriented detector-score change to paired transformation-output measurements. Raw and BH-adjusted p-values for all 48 correlations are retained in the distance/diagnostic CSV; the table reports descriptive coefficients only.\n",
        "| detector | orthographic density change | morphosyntactic density change | executed rules | word-count change |",
        "|---|---:|---:|---:|---:|",
    ]
    for detector in ZERO_SHOT_DETECTORS:
        row = detector_pivot.loc[detector]
        lines.append(
            f"| {detector} | {row['paired_orthographic_density_change']:+.3f} | "
            f"{row['paired_morphosyntactic_density_change']:+.3f} | "
            f"{row['paired_executed_rule_count']:+.3f} | {row['paired_word_count_change']:+.3f} |"
        )
    lines.append("\nThese correlations are observational properties of transformation outputs and do not identify which linguistic changes caused detector-score shifts. They are not causal mediators.\n")

    lines += [
        "## L. Claim audit\n",
        "| claim | assessment | evidence and scope |",
        "|---|---|---|",
        f"| **A. The rule transformation moves SAE toward natural AAVE on the measured dialect features.** | **NOT SUPPORTED** | The predeclared primary L1 ratio is {raw['l1_distance_ratio']:.6f}, with distance change {raw['l1_distance_change']:+.8f}; aggregate densities and some features move closer, but the full measured-feature centroid does not. |",
        "| **B. The transformed profile reproduces the natural AAVE feature distribution.** | **NOT SUPPORTED** | Moving selected summaries is insufficient, and the individual-feature profile remains far from the natural distribution. |",
        "| **C. The proxy's detector behavior can be explained by failure to introduce measured dialect features.** | **NOT SUPPORTED** | The transformation substantially changes measured morphosyntactic features, and the exploratory correlations do not identify causes. |",
        "| **D. The proxy can move toward natural AAVE linguistically under measured features while nevertheless producing detector behavior that differs from the natural audit.** | **SUPPORTED WITH QUALIFICATION** | Both aggregate-density mean gaps narrow and selected features move toward the natural mean, while Steps 6-7 show detector-audit divergence. The primary multivariate L1 profile does not move closer, so this applies to specific measured summaries rather than overall profile fidelity. |",
        "| **E. The measured dialect-profile result generalizes to full linguistic authenticity.** | **NOT SUPPORTED** | The measurements cover 25 orthographic forms and 10 regex-based morphosyntactic features and cannot establish authenticity. |",
        "\n## M. Final decision\n",
    ]
    if final_label == "PROCEED TO INTEGRATED PROVENANCE SYNTHESIS":
        lines.append("The primary measured-feature distance moves closer with bootstrap support, while the report retains feature-level and source-mixture qualifications.\n")
    else:
        lines.append("Aggregate densities move closer, but the predeclared multivariate L1 profile does not. The measured-feature evidence therefore cannot validate the rule-transformed arm as a natural-AAVE proxy.\n")
    lines.append(f"**{final_label}**")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    aave, sae, canonical, paired_scores = load_and_validate()
    profiles = build_profiles(aave, sae, canonical)
    features, distances, results = analyze(profiles, paired_scores)
    profiles.to_csv(PROFILES_OUT, index=False, lineterminator="\n")
    features.to_csv(FEATURE_OUT, index=False, lineterminator="\n")
    distances.to_csv(DISTANCE_OUT, index=False, lineterminator="\n")
    make_figure(features)
    write_report(results)
    raw = results["distances"]["unstandardized"]
    print({
        "documents": len(profiles),
        "features": len(features),
        "l1_original": raw["l1_d_original"],
        "l1_transformed": raw["l1_d_transformed"],
        "distance_ratio": raw["l1_distance_ratio"],
        "bootstrap_p_change_lt_zero": raw["bootstrap_proportion_distance_change_lt_zero"],
    })
    for path in [REPORT, PROFILES_OUT, FEATURE_OUT, DISTANCE_OUT, FIGURE]:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""RQ1 proxy Step 5: within-poem paired transformation effect.

Estimand: transformed detector score - original SAE detector score, for the
SAME poem (the canonical Multi-VALUE-transformed SAE arm vs. the frozen
Step 1 original SAE score for that exact poem). This is NOT the natural
AAVE-vs-SAE population comparison; it is a paired, within-poem intervention
effect. Consumes only frozen Step 1-4 outputs; no detector inference, no
GPTZero call, no manuscript edit, no natural-vs-synthetic comparison, no
threshold sweep, no proxy-fidelity correlation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, wilcoxon, binomtest, chi2 as chi2_dist

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data/analysis"
FIGURES = ROOT / "figures"
SEED = 20260918  # same fixed seed used by the natural RQ1 (Step 1) analysis
N_BOOT = 10_000

PAIRED_PATH = ANALYSIS / "rq1_proxy_original_canonical_paired_scores.csv"
CANONICAL_SCORES_PATH = ANALYSIS / "rq1_proxy_canonical_detector_scores.csv"
NATURAL_PATH = ANALYSIS / "rq1_full_poetry_natural_scores.csv"
CANONICAL_ARM_PATH = ANALYSIS / "rq1_proxy_sae521_multivalue_canonical.csv"
SENSITIVITY_PATH = ANALYSIS / "rq1_proxy_fresh_instance_sensitivity_scores.csv"

ZERO_SHOT_DETECTORS = [
    "Log Probability (GPT2-XL)", "Rank (negated) (GPT2-XL)", "Log-Rank (negated) (GPT2-XL)",
    "Entropy (GPT2-XL)", "DetectLLM-LRR (GPT2-XL)", "Binoculars (GPT2-XL)",
    "Log Probability (OPT-1.3B)", "Rank (negated) (OPT-1.3B)", "Log-Rank (negated) (OPT-1.3B)",
    "Entropy (OPT-1.3B)", "DetectLLM-LRR (OPT-1.3B)", "Binoculars (OPT-1.3B)",
]
PRETRAINED_DETECTORS = ["OpenAI-RoBERTa-base", "OpenAI-RoBERTa-large", "AIGC-MPU"]


def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


# --------------------------------------------------------------------------- #
# 1. Load + validate
# --------------------------------------------------------------------------- #

def load_and_validate() -> dict:
    paired = pd.read_csv(PAIRED_PATH)
    canon = pd.read_csv(CANONICAL_SCORES_PATH)
    natural = pd.read_csv(NATURAL_PATH)
    natural_sae = natural.loc[natural["variety"].eq("SAE")].copy()
    canon_arm = pd.read_csv(CANONICAL_ARM_PATH)

    checks = {}
    checks["n_unique_source_poems_521"] = int(paired["source_doc_id"].nunique()) == 521
    checks["n_detector_configs_15"] = int(paired["detector"].nunique()) == 15
    checks["n_rows_7815"] = len(paired) == 7815
    checks["no_missing_original_scores"] = int(paired["original_sae_score"].isna().sum()) == 0
    checks["no_missing_transformed_scores"] = int(paired["canonical_transformed_score"].isna().sum()) == 0
    checks["no_duplicate_detector_source_pairs"] = not paired.duplicated(["detector", "source_doc_id"]).any()

    hash_map = canon_arm.set_index("source_doc_id")["transformed_sha256"]
    checks["transformed_hashes_match_step3_canonical_arm"] = bool(
        paired["transformed_sha256"].eq(paired["source_doc_id"].map(hash_map)).all()
    )

    nat_raw = natural_sae.set_index(["source_doc_id", "detector"])["raw_score"]
    joined_raw = paired.set_index(["source_doc_id", "detector"])["original_sae_raw_score"]
    checks["original_sae_scores_match_step1"] = bool(np.allclose(
        joined_raw.reindex(nat_raw.index).to_numpy(), nat_raw.to_numpy(), rtol=0, atol=0, equal_nan=False
    )) if len(joined_raw) == len(nat_raw) else False

    nat_fold = natural_sae.drop_duplicates("source_doc_id").set_index("source_doc_id")["evaluation_fold"]
    paired_fold = paired.drop_duplicates("source_doc_id").set_index("source_doc_id")["sae_fold"]
    checks["fold_assignments_match_step1"] = bool((paired_fold.reindex(nat_fold.index) == nat_fold).all())

    nat_orient = natural_sae.drop_duplicates("detector").set_index("detector")["orientation_multiplier"]
    canon_orient = canon.drop_duplicates("detector").set_index("detector")["orientation_multiplier"]
    checks["detector_orientation_matches_step1"] = bool((canon_orient.reindex(nat_orient.index) == nat_orient).all())

    if not all(checks.values()):
        failed = {k: v for k, v in checks.items() if not v}
        raise AssertionError(f"Pairing validation failed: {failed}")

    return {"paired": paired, "canon": canon, "natural_sae": natural_sae, "canon_arm": canon_arm, "checks": checks}


# --------------------------------------------------------------------------- #
# 2-6. Per-detector paired statistics
# --------------------------------------------------------------------------- #

def rank_biserial_point(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    res = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    u = float(res.statistic)
    r = 2.0 * u / (len(x) * len(y)) - 1.0
    return u, float(res.pvalue), r


def rank_biserial_bootstrap(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    xb = x[idx]
    yb = y[idx]
    combined = np.concatenate([xb, yb], axis=1)
    ranks = rankdata(combined, method="average", axis=1)
    rx = ranks[:, :n].sum(axis=1)
    u = rx - n * (n + 1) / 2.0
    return 2.0 * u / (n * n) - 1.0


def mcnemar(b: int, c: int) -> tuple[float, float, str]:
    if b + c == 0:
        return float("nan"), 1.0, "no_discordant_pairs"
    if b + c < 25:
        p = binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
        return float("nan"), float(p), "exact_binomial(b+c<25)"
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(chi2_dist.sf(stat, df=1))
    return float(stat), p, "continuity_corrected_chi2(b+c>=25)"


def per_detector_stats(detector: str, sub: pd.DataFrame, idx: np.ndarray) -> dict:
    sub = sub.sort_values("source_doc_id")
    original = sub["original_sae_score"].to_numpy(float)
    transformed = sub["canonical_transformed_score"].to_numpy(float)
    diff = transformed - original
    n = len(diff)
    assert n == 521

    diff_b = diff[idx]
    mean_b = diff_b.mean(axis=1)
    median_b = np.median(diff_b, axis=1)
    mean_ci = np.quantile(mean_b, [0.025, 0.975])
    median_ci = np.quantile(median_b, [0.025, 0.975])

    q1, q3 = np.percentile(diff, [25, 75])
    n_up, n_down, n_eq = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())

    wres = wilcoxon(transformed, original, zero_method="wilcox", alternative="two-sided", method="approx")
    n_zero_dropped = n_eq  # 'wilcox' zero_method drops exact-zero-difference pairs before ranking
    if n_up + n_down > 0:
        sign_p = binomtest(n_up, n_up + n_down, 0.5, alternative="two-sided").pvalue
    else:
        sign_p = float("nan")

    u, mw_p, r = rank_biserial_point(transformed, original)
    r_boot = rank_biserial_bootstrap(transformed, original, idx)
    r_ci = np.quantile(r_boot, [0.025, 0.975])

    orig_flag = sub["original_flag_machine"].astype(bool).to_numpy()
    trans_flag = sub["canonical_flag_machine"].astype(bool).to_numpy()
    fpr_original = float(orig_flag.mean())
    fpr_transformed = float(trans_flag.mean())
    delta = fpr_transformed - fpr_original
    b_flip = int((~orig_flag & trans_flag).sum())  # original human -> transformed machine
    c_flip = int((orig_flag & ~trans_flag).sum())  # original machine -> transformed human
    mcn_stat, mcn_p, mcn_method = mcnemar(b_flip, c_flip)

    orig_flag_b = orig_flag[idx]
    trans_flag_b = trans_flag[idx]
    delta_b = trans_flag_b.mean(axis=1) - orig_flag_b.mean(axis=1)
    delta_ci = np.quantile(delta_b, [0.025, 0.975])

    family = sub["detector"].map(lambda d: "zero-shot" if d in ZERO_SHOT_DETECTORS else "pretrained").iloc[0]

    return {
        "detector": detector, "family": family, "n_pairs": n,
        "mean_original": float(original.mean()), "median_original": float(np.median(original)),
        "mean_transformed": float(transformed.mean()), "median_transformed": float(np.median(transformed)),
        "mean_paired_change": float(diff.mean()), "median_paired_change": float(np.median(diff)),
        "mean_paired_change_ci_lo": float(mean_ci[0]), "mean_paired_change_ci_hi": float(mean_ci[1]),
        "median_paired_change_ci_lo": float(median_ci[0]), "median_paired_change_ci_hi": float(median_ci[1]),
        "std_paired_change": float(diff.std(ddof=1)), "iqr_paired_change": float(q3 - q1),
        "n_up": n_up, "n_down": n_down, "n_equal": n_eq,
        "pct_up": 100.0 * n_up / n, "pct_down": 100.0 * n_down / n, "pct_equal": 100.0 * n_eq / n,
        "wilcoxon_statistic": float(wres.statistic), "wilcoxon_p_raw": float(wres.pvalue),
        "wilcoxon_zero_method": "wilcox (zero-difference pairs dropped before ranking)",
        "wilcoxon_n_zero_dropped": n_zero_dropped,
        "sign_test_p_raw": float(sign_p),
        "mann_whitney_u": u, "mann_whitney_p": mw_p,
        "rank_biserial_r": r, "rank_biserial_ci_lo": float(r_ci[0]), "rank_biserial_ci_hi": float(r_ci[1]),
        "original_sae_fpr": fpr_original, "transformed_synthetic_fpr": fpr_transformed,
        "synthetic_operational_delta": delta,
        "synthetic_operational_delta_ci_lo": float(delta_ci[0]), "synthetic_operational_delta_ci_hi": float(delta_ci[1]),
        "b_human_to_machine": b_flip, "c_machine_to_human": c_flip,
        "mcnemar_statistic": mcn_stat, "mcnemar_p_raw": mcn_p, "mcnemar_method": mcn_method,
    }


def run_primary(paired: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 521, size=(N_BOOT, 521))
    detectors = sorted(paired["detector"].unique())
    rows = [per_detector_stats(d, paired.loc[paired["detector"].eq(d)], idx) for d in detectors]
    out = pd.DataFrame(rows)
    out["wilcoxon_p_bh_15"] = bh_adjust(out["wilcoxon_p_raw"].to_numpy())
    out["sign_test_p_bh_15"] = bh_adjust(out["sign_test_p_raw"].to_numpy())
    out["mcnemar_p_bh_15"] = bh_adjust(out["mcnemar_p_raw"].to_numpy())
    return out, idx


# --------------------------------------------------------------------------- #
# 9. Persistent-state (fresh-instance) robustness
# --------------------------------------------------------------------------- #

def run_fresh_instance(paired: pd.DataFrame, sensitivity: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    orig = paired[["source_doc_id", "detector", "original_sae_score", "original_flag_machine"]]
    sens = sensitivity[["source_doc_id", "detector", "sensitivity_oriented_score", "sensitivity_flag_machine"]]
    merged = orig.merge(sens, on=["source_doc_id", "detector"], how="inner", validate="one_to_one")
    assert len(merged) == 7815
    merged = merged.rename(columns={
        "sensitivity_oriented_score": "canonical_transformed_score",
        "sensitivity_flag_machine": "canonical_flag_machine",
    })
    detectors = sorted(merged["detector"].unique())
    rows = [per_detector_stats(d, merged.loc[merged["detector"].eq(d)], idx) for d in detectors]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 10. Length-change diagnostic
# --------------------------------------------------------------------------- #

def length_diagnostic(paired: pd.DataFrame, canon_arm: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr
    lengths = canon_arm[["source_doc_id", "original_word_count", "transformed_word_count",
                          "original_character_count", "transformed_character_count"]].copy()
    lengths["word_count_change"] = lengths["transformed_word_count"] - lengths["original_word_count"]
    lengths["character_count_change"] = lengths["transformed_character_count"] - lengths["original_character_count"]
    merged = paired.merge(lengths[["source_doc_id", "word_count_change", "character_count_change"]],
                           on="source_doc_id", how="left", validate="many_to_one")
    rows = []
    for detector, sub in merged.groupby("detector", sort=True):
        diff = sub["paired_change"].to_numpy(float)
        rho_w, p_w = spearmanr(diff, sub["word_count_change"].to_numpy(float))
        rho_c, p_c = spearmanr(diff, sub["character_count_change"].to_numpy(float))
        rows.append({
            "detector": detector,
            "spearman_rho_word_count_change": float(rho_w), "spearman_p_word_count_change": float(p_w),
            "spearman_rho_character_count_change": float(rho_c), "spearman_p_character_count_change": float(p_c),
        })
    out = pd.DataFrame(rows)
    out["p_bh_15_word"] = bh_adjust(out["spearman_p_word_count_change"].to_numpy())
    out["p_bh_15_char"] = bh_adjust(out["spearman_p_character_count_change"].to_numpy())
    return out


# --------------------------------------------------------------------------- #
# 11. Figure
# --------------------------------------------------------------------------- #

def make_figure(paired: pd.DataFrame, idx: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for detector in ZERO_SHOT_DETECTORS:
        sub = paired.loc[paired["detector"].eq(detector)].sort_values("source_doc_id")
        original = sub["original_sae_score"].to_numpy(float)
        transformed = sub["canonical_transformed_score"].to_numpy(float)
        diff = transformed - original
        sd = diff.std(ddof=1)  # unused; normalization uses original SD per spec
        normalized_point = diff.mean() / original.std(ddof=1)

        diff_b = diff[idx]
        original_b = original[idx]
        normalized_b = diff_b.mean(axis=1) / original_b.std(axis=1, ddof=1)
        lo, hi = np.quantile(normalized_b, [0.025, 0.975])
        rows.append({"detector": detector, "point": normalized_point, "lo": lo, "hi": hi})

    fig_df = pd.DataFrame(rows).sort_values("point")
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(fig_df))
    colors = ["#4C72B0" if "GPT2" in d else "#DD8452" for d in fig_df["detector"]]
    ax.errorbar(
        fig_df["point"], y_pos,
        xerr=[fig_df["point"] - fig_df["lo"], fig_df["hi"] - fig_df["point"]],
        fmt="o", ecolor="gray", capsize=3, color="black", zorder=3,
    )
    for i, c in enumerate(colors):
        ax.plot(fig_df["point"].iloc[i], y_pos[i], "o", color=c, zorder=4)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fig_df["detector"])
    ax.set_xlabel("Normalized paired transformation effect\n(mean[transformed - original] / SD[original SAE score]), with 95% paired bootstrap CI")
    ax.set_title("RQ1 proxy: within-poem paired transformation effect, 12 zero-shot configurations\n(all inferential analyses remain on raw detector scales; this plot is a normalized display only)")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "rq1_proxy_paired_transformation_effect.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    data = load_and_validate()
    paired, canon_arm = data["paired"], data["canon_arm"]

    primary, idx = run_primary(paired)
    primary.to_csv(ANALYSIS / "rq1_proxy_paired_effects.csv", index=False)

    operational_cols = [
        "detector", "family", "original_sae_fpr", "transformed_synthetic_fpr", "synthetic_operational_delta",
        "synthetic_operational_delta_ci_lo", "synthetic_operational_delta_ci_hi",
        "b_human_to_machine", "c_machine_to_human", "mcnemar_statistic", "mcnemar_p_raw",
        "mcnemar_p_bh_15", "mcnemar_method",
    ]
    primary[operational_cols].to_csv(ANALYSIS / "rq1_proxy_operational_paired_effects.csv", index=False)

    sensitivity = pd.read_csv(SENSITIVITY_PATH)
    fresh = run_fresh_instance(paired, sensitivity, idx)
    fresh.to_csv(ANALYSIS / "rq1_proxy_paired_effects_fresh_instance.csv", index=False)

    length_diag = length_diagnostic(paired, canon_arm)
    length_diag.to_csv(ANALYSIS / "rq1_proxy_paired_length_diagnostic.csv", index=False)

    make_figure(paired, idx)

    # comparison of canonical vs fresh-instance sign/CI-crossing
    cmp_rows = []
    for detector in primary["detector"]:
        c = primary.loc[primary["detector"].eq(detector)].iloc[0]
        f = fresh.loc[fresh["detector"].eq(detector)].iloc[0]
        cmp_rows.append({
            "detector": detector,
            "mean_change_sign_flip": bool(np.sign(c["mean_paired_change"]) != np.sign(f["mean_paired_change"])),
            "median_change_sign_flip": bool(np.sign(c["median_paired_change"]) != np.sign(f["median_paired_change"])),
            "rank_biserial_sign_flip": bool(np.sign(c["rank_biserial_r"]) != np.sign(f["rank_biserial_r"])),
            "operational_delta_sign_flip": bool(np.sign(c["synthetic_operational_delta"]) != np.sign(f["synthetic_operational_delta"])),
            "mean_change_ci_excl_zero_canonical": bool(c["mean_paired_change_ci_lo"] > 0 or c["mean_paired_change_ci_hi"] < 0),
            "mean_change_ci_excl_zero_fresh": bool(f["mean_paired_change_ci_lo"] > 0 or f["mean_paired_change_ci_hi"] < 0),
            "rank_biserial_ci_excl_zero_canonical": bool(c["rank_biserial_ci_lo"] > 0 or c["rank_biserial_ci_hi"] < 0),
            "rank_biserial_ci_excl_zero_fresh": bool(f["rank_biserial_ci_lo"] > 0 or f["rank_biserial_ci_hi"] < 0),
            "operational_delta_ci_excl_zero_canonical": bool(c["synthetic_operational_delta_ci_lo"] > 0 or c["synthetic_operational_delta_ci_hi"] < 0),
            "operational_delta_ci_excl_zero_fresh": bool(f["synthetic_operational_delta_ci_lo"] > 0 or f["synthetic_operational_delta_ci_hi"] < 0),
        })
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(ANALYSIS / "rq1_proxy_paired_canonical_vs_fresh_instance_comparison.csv", index=False)

    summary = {
        "seed": SEED, "n_boot": N_BOOT,
        "validation_checks": data["checks"],
        "zero_shot_summary": {
            "n_positive_median_change": int((primary.loc[primary.detector.isin(ZERO_SHOT_DETECTORS), "median_paired_change"] > 0).sum()),
            "n_negative_median_change": int((primary.loc[primary.detector.isin(ZERO_SHOT_DETECTORS), "median_paired_change"] < 0).sum()),
            "n_median_ci_excl_zero_positive": int(((primary.detector.isin(ZERO_SHOT_DETECTORS)) & (primary["median_paired_change_ci_lo"] > 0)).sum()),
            "n_median_ci_excl_zero_negative": int(((primary.detector.isin(ZERO_SHOT_DETECTORS)) & (primary["median_paired_change_ci_hi"] < 0)).sum()),
            "n_positive_r": int((primary.loc[primary.detector.isin(ZERO_SHOT_DETECTORS), "rank_biserial_r"] > 0).sum()),
            "n_negative_r": int((primary.loc[primary.detector.isin(ZERO_SHOT_DETECTORS), "rank_biserial_r"] < 0).sum()),
            "n_r_ci_excl_zero_positive": int(((primary.detector.isin(ZERO_SHOT_DETECTORS)) & (primary["rank_biserial_ci_lo"] > 0)).sum()),
            "n_r_ci_excl_zero_negative": int(((primary.detector.isin(ZERO_SHOT_DETECTORS)) & (primary["rank_biserial_ci_hi"] < 0)).sum()),
            "n_positive_delta": int((primary.loc[primary.detector.isin(ZERO_SHOT_DETECTORS), "synthetic_operational_delta"] > 0).sum()),
            "n_negative_delta": int((primary.loc[primary.detector.isin(ZERO_SHOT_DETECTORS), "synthetic_operational_delta"] < 0).sum()),
            "n_delta_ci_excl_zero_positive": int(((primary.detector.isin(ZERO_SHOT_DETECTORS)) & (primary["synthetic_operational_delta_ci_lo"] > 0)).sum()),
            "n_delta_ci_excl_zero_negative": int(((primary.detector.isin(ZERO_SHOT_DETECTORS)) & (primary["synthetic_operational_delta_ci_hi"] < 0)).sum()),
            "n_bh_significant_mcnemar": int(((primary.detector.isin(ZERO_SHOT_DETECTORS)) & (primary["mcnemar_p_bh_15"] < 0.05)).sum()),
        },
        "pretrained_bh_significant_mcnemar": int(((primary.detector.isin(PRETRAINED_DETECTORS)) & (primary["mcnemar_p_bh_15"] < 0.05)).sum()),
    }
    (ANALYSIS / "rq1_proxy_paired_transformation_effect_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("wrote rq1_proxy_paired_effects.csv, rq1_proxy_operational_paired_effects.csv,")
    print("      rq1_proxy_paired_effects_fresh_instance.csv, rq1_proxy_paired_length_diagnostic.csv,")
    print("      rq1_proxy_paired_canonical_vs_fresh_instance_comparison.csv, figures/rq1_proxy_paired_transformation_effect.pdf")


if __name__ == "__main__":
    main()

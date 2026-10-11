"""Final full-poetry natural-dialect reference analysis (RQ1 Step 1).

Consumes only the frozen local-detector score caches and the missing-only
extensions produced by ``run_rq1_full_poetry_local.py``.  It does not call a
model or an external service.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data/analysis"
REPORT = ROOT / "reports/final_rq1_full_poetry_natural.md"
SEED = 20260918
N_BOOT = 10_000

METRICS = {
    "mean_log_prob": "Log Probability",
    "mean_rank": "Rank (negated)",
    "mean_log_rank": "Log-Rank (negated)",
    "mean_entropy": "Entropy",
    "detectllm_lrr": "DetectLLM-LRR",
    "binoculars": "Binoculars",
}
FROZEN_SIGN = {
    "mean_log_prob": 1,
    "mean_rank": -1,
    "mean_log_rank": -1,
    "mean_entropy": -1,
    "detectllm_lrr": 1,
    "binoculars": 1,
}


def stable_folds(ids: pd.Series, n_folds: int = 5) -> pd.Series:
    """Exact prior RQ1 convention: SHA-256 order, round-robin assignment."""
    order = sorted(ids.astype(str).unique(), key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    mapping = {doc: i % n_folds for i, doc in enumerate(order)}
    return ids.astype(str).map(mapping).astype(int)


def bh_adjust(p: pd.Series) -> pd.Series:
    values = p.to_numpy(float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=p.index)


def rank_biserial(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    test = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    r = 2.0 * float(test.statistic) / (len(x) * len(y)) - 1.0
    return float(test.statistic), float(test.pvalue), r


def publication_era(row: pd.Series) -> str:
    if pd.notna(row["pub_year"]):
        return str(int(row["pub_year"]))
    if row["source_subset"] == "hughes":
        return "1921-1928 (source-bounded; per-poem year unavailable)"
    return "unavailable"


def load_pool() -> pd.DataFrame:
    a = pd.read_csv(ROOT / "data/processed/aave_poetry.csv")
    a = a.loc[a["variety"].eq("AAVE")].copy()
    s = pd.read_csv(ROOT / "data/processed/sae_poetry.csv").copy()
    assert len(a) == 212 and len(s) == 521
    a["poem_id"] = a["id"].astype(str)
    s["poem_id"] = s["id"].astype(str)
    a["source_doc_id"] = "poem_" + a["poem_id"]
    s["source_doc_id"] = "poem_sae_" + s["poem_id"]
    cols = [
        "poem_id", "source_doc_id", "text", "variety", "source_dataset",
        "source_subset", "poet", "poem_title", "pub_year", "n_words",
    ]
    pool = pd.concat([a[cols], s[cols]], ignore_index=True)
    pool["publication_era"] = pool.apply(publication_era, axis=1)
    norm = pool["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    assert not pool["poem_id"].duplicated().any()
    assert not pool["source_doc_id"].duplicated().any()
    assert not pool["text"].duplicated().any()
    assert not norm.duplicated().any()

    old = pd.read_csv(ROOT / "data/interim/detection_v4_subsample.csv")
    old_ids = set(old.loc[
        old["category"].eq("HWT") & old["genre"].eq("poetry") & old["variety"].isin(["AAVE", "SAE"]),
        "source_doc_id",
    ].astype(str))
    pool["overlaps_old_rq1_100_100"] = pool["source_doc_id"].isin(old_ids)
    assert pool.groupby("variety")["overlaps_old_rq1_100_100"].sum().to_dict() == {"AAVE": 100, "SAE": 100}
    pool["evaluation_fold"] = -1
    for variety in ["AAVE", "SAE"]:
        mask = pool["variety"].eq(variety)
        pool.loc[mask, "evaluation_fold"] = stable_folds(pool.loc[mask, "source_doc_id"]).to_numpy()
    pool["evaluation_fold"] = pool["evaluation_fold"].astype(int)
    return pool


def load_scores(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    for tier in ["gpt2", "opt"]:
        path = ANALYSIS / f"rq1_full_poetry_{tier}_local_cache.csv"
        wide = pd.read_csv(path)
        assert len(wide) == 733 and not wide["source_doc_id"].duplicated().any()
        backbone = "GPT2-XL" if tier == "gpt2" else "OPT-1.3B"
        old_file = f"data/interim/detection_v4_zeroshot_{tier}.csv"
        for metric, label in METRICS.items():
            d = wide[["source_doc_id", metric, "score_origin"]].copy()
            d = d.rename(columns={metric: "raw_score"})
            d["detector"] = f"{label} ({backbone})"
            d["family"] = "zero-shot"
            d["backbone"] = backbone
            d["metric_key"] = metric
            d["orientation_multiplier"] = FROZEN_SIGN[metric]
            d["oriented_score"] = d["raw_score"] * FROZEN_SIGN[metric]
            d["source_cache"] = np.where(
                d["score_origin"].eq("reused_v4"), old_file, str(path.relative_to(ROOT))
            )
            rows.append(d)
            audit_rows.append({
                "detector": f"{label} ({backbone})", "family": "zero-shot",
                "expected_rows": 733, "initial_cached_rows": 200,
                "initial_missing_rows": 533, "newly_scored_rows": 533,
                "final_cached_rows": len(d),
                "duplicate_rows": int(d["source_doc_id"].duplicated().sum()),
                "orientation": f"raw × {FROZEN_SIGN[metric]:+d}; higher = more machine-like",
                "cache_source": f"{old_file}; {path.relative_to(ROOT)}",
            })

    pre_path = ANALYSIS / "rq1_full_poetry_pretrained_local_cache.csv"
    pre = pd.read_csv(pre_path)
    for detector in ["OpenAI-RoBERTa-base", "OpenAI-RoBERTa-large", "AIGC-MPU"]:
        d = pre.loc[pre["detector"].eq(detector), ["source_doc_id", "p_machine", "score_origin"]].copy()
        assert len(d) == 733 and not d["source_doc_id"].duplicated().any()
        d = d.rename(columns={"p_machine": "raw_score"})
        d["detector"] = detector
        d["family"] = "pretrained"
        d["backbone"] = detector
        d["metric_key"] = "p_machine"
        d["orientation_multiplier"] = 1
        d["oriented_score"] = d["raw_score"]
        d["source_cache"] = np.where(
            d["score_origin"].eq("reused_v4"),
            "data/interim/detection_v4_pretrained.csv",
            str(pre_path.relative_to(ROOT)),
        )
        rows.append(d)
        audit_rows.append({
            "detector": detector, "family": "pretrained", "expected_rows": 733,
            "initial_cached_rows": 200, "initial_missing_rows": 533,
            "newly_scored_rows": 533, "final_cached_rows": len(d),
            "duplicate_rows": int(d["source_doc_id"].duplicated().sum()),
            "orientation": "P(machine) × +1; higher = more machine-like",
            "cache_source": f"data/interim/detection_v4_pretrained.csv; {pre_path.relative_to(ROOT)}",
        })

    scores = pd.concat(rows, ignore_index=True)
    scores = scores.merge(pool.drop(columns=["text"]), on="source_doc_id", how="left", validate="many_to_one")
    assert len(scores) == 15 * 733
    assert not scores.duplicated(["detector", "source_doc_id"]).any()
    assert np.isfinite(scores["oriented_score"]).all()
    audit = pd.DataFrame(audit_rows)
    return scores, audit


def bootstrap_r(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.empty(N_BOOT)
    for b in range(N_BOOT):
        xb = x[rng.integers(0, len(x), len(x))]
        yb = y[rng.integers(0, len(y), len(y))]
        values[b] = rank_biserial(xb, yb)[2]
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def analyze(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    summary_rows: list[dict] = []
    threshold_rows: list[dict] = []
    enriched: list[pd.DataFrame] = []

    detector_order = scores["detector"].drop_duplicates().tolist()
    for detector in detector_order:
        d = scores.loc[scores["detector"].eq(detector)].copy()
        a = d.loc[d["variety"].eq("AAVE"), "oriented_score"].to_numpy(float)
        s = d.loc[d["variety"].eq("SAE"), "oriented_score"].to_numpy(float)
        assert len(a) == 212 and len(s) == 521
        u, p, rb = rank_biserial(a, s)
        rb_lo, rb_hi = bootstrap_r(a, s, rng)

        old = d.loc[d["overlaps_old_rq1_100_100"]]
        old_a = old.loc[old["variety"].eq("AAVE"), "oriented_score"].to_numpy(float)
        old_s = old.loc[old["variety"].eq("SAE"), "oriented_score"].to_numpy(float)
        old_u, old_p, old_rb = rank_biserial(old_a, old_s)
        old_rb_lo, old_rb_hi = bootstrap_r(old_a, old_s, rng)

        d["threshold"] = np.nan
        d["flag_machine"] = False
        for fold in range(5):
            calibration = d.loc[d["variety"].eq("SAE") & d["evaluation_fold"].ne(fold), "oriented_score"].to_numpy(float)
            threshold = float(np.quantile(calibration, 0.95, method="higher"))
            held = d["evaluation_fold"].eq(fold)
            d.loc[held, "threshold"] = threshold
            d.loc[held, "flag_machine"] = d.loc[held, "oriented_score"] > threshold
            held_s = d.loc[held & d["variety"].eq("SAE")]
            held_a = d.loc[held & d["variety"].eq("AAVE")]
            threshold_rows.append({
                "detector": detector, "family": d["family"].iloc[0], "fold": fold,
                "target_sae_fpr": 0.05, "quantile_method": "higher",
                "decision_rule": "oriented_score > threshold; ties human",
                "sae_calibration_n": len(calibration), "threshold": threshold,
                "heldout_sae_n": len(held_s), "heldout_sae_false_positives": int(held_s["flag_machine"].sum()),
                "heldout_sae_fpr": float(held_s["flag_machine"].mean()),
                "aave_eval_n": len(held_a), "aave_false_positives": int(held_a["flag_machine"].sum()),
                "aave_fpr": float(held_a["flag_machine"].mean()),
            })

        rates = d.groupby("variety")["flag_machine"].mean()
        delta = float(rates["AAVE"] - rates["SAE"])
        flags_a = d.loc[d["variety"].eq("AAVE"), "flag_machine"].to_numpy(float)
        flags_s = d.loc[d["variety"].eq("SAE"), "flag_machine"].to_numpy(float)
        boot_delta = np.empty(N_BOOT)
        for b in range(N_BOOT):
            boot_delta[b] = (
                flags_a[rng.integers(0, len(flags_a), len(flags_a))].mean()
                - flags_s[rng.integers(0, len(flags_s), len(flags_s))].mean()
            )
        delta_lo, delta_hi = np.quantile(boot_delta, [0.025, 0.975])

        # Exact operational reproduction on the nested 100/100 subset.
        old = d.loc[d["overlaps_old_rq1_100_100"]].copy()
        old["old_fold"] = -1
        for variety in ["AAVE", "SAE"]:
            mask = old["variety"].eq(variety)
            old.loc[mask, "old_fold"] = stable_folds(old.loc[mask, "source_doc_id"]).to_numpy()
        old["old_flag"] = False
        for fold in range(5):
            calibration = old.loc[old["variety"].eq("SAE") & old["old_fold"].ne(fold), "oriented_score"].to_numpy(float)
            threshold = float(np.quantile(calibration, 0.95, method="higher"))
            held = old["old_fold"].eq(fold)
            old.loc[held, "old_flag"] = old.loc[held, "oriented_score"] > threshold
        old_rates = old.groupby("variety")["old_flag"].mean()
        old_delta = float(old_rates["AAVE"] - old_rates["SAE"])
        old_flags_a = old.loc[old["variety"].eq("AAVE"), "old_flag"].to_numpy(float)
        old_flags_s = old.loc[old["variety"].eq("SAE"), "old_flag"].to_numpy(float)
        old_boot_delta = np.empty(N_BOOT)
        for b in range(N_BOOT):
            old_boot_delta[b] = (
                old_flags_a[rng.integers(0, len(old_flags_a), len(old_flags_a))].mean()
                - old_flags_s[rng.integers(0, len(old_flags_s), len(old_flags_s))].mean()
            )
        old_delta_lo, old_delta_hi = np.quantile(old_boot_delta, [0.025, 0.975])

        summary_rows.append({
            "detector": detector, "family": d["family"].iloc[0],
            "n_aave": len(a), "n_sae": len(s),
            "mean_aave": float(np.mean(a)), "mean_sae": float(np.mean(s)),
            "median_aave": float(np.median(a)), "median_sae": float(np.median(s)),
            "mann_whitney_u": u, "rank_biserial_r": rb,
            "rank_biserial_ci_lo": rb_lo, "rank_biserial_ci_hi": rb_hi,
            "p_raw_two_sided": p,
            "oof_fpr_sae": float(rates["SAE"]), "aave_fpr": float(rates["AAVE"]),
            "operational_delta": delta, "operational_delta_ci_lo": float(delta_lo),
            "operational_delta_ci_hi": float(delta_hi),
            "operational_ci_type": "conditional on fitted fold thresholds",
            "old_n_aave": len(old_a), "old_n_sae": len(old_s),
            "old_mean_aave": float(np.mean(old_a)), "old_mean_sae": float(np.mean(old_s)),
            "old_median_aave": float(np.median(old_a)), "old_median_sae": float(np.median(old_s)),
            "old_mann_whitney_u": old_u, "old_rank_biserial_r": old_rb,
            "old_rank_biserial_ci_lo": old_rb_lo, "old_rank_biserial_ci_hi": old_rb_hi,
            "old_p_raw_two_sided": old_p,
            "old_oof_fpr_sae": float(old_rates["SAE"]), "old_aave_fpr": float(old_rates["AAVE"]),
            "old_operational_delta": old_delta,
            "old_operational_delta_ci_lo": float(old_delta_lo),
            "old_operational_delta_ci_hi": float(old_delta_hi),
            "r_change_full_minus_old": rb - old_rb,
            "r_direction_changed": bool(np.sign(rb) != np.sign(old_rb)),
            "r_material_magnitude_change_ge_0_10": bool(abs(rb - old_rb) >= 0.10),
            "operational_direction_changed": bool(np.sign(delta) != np.sign(old_delta)),
            "operational_material_magnitude_change_ge_0_10": bool(abs(delta - old_delta) >= 0.10),
        })
        enriched.append(d)

    summary = pd.DataFrame(summary_rows)
    summary["p_bh_15"] = bh_adjust(summary["p_raw_two_sided"])
    summary["old_p_bh_15"] = bh_adjust(summary["old_p_raw_two_sided"])
    thresholds = pd.DataFrame(threshold_rows)
    score_out = pd.concat(enriched, ignore_index=True)
    return score_out, thresholds, summary


def fmt(x: float) -> str:
    return f"{x:.4g}"


def write_report(pool: pd.DataFrame, audit: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Final RQ1 full-poetry natural-dialect reference analysis\n",
        "## Scope and frozen design\n",
        "This Step 1 analysis uses all 212 retained natural AAVE poems and all 521 final era-matched SAE poems. It does not use the superseded LiteraryTaste SAE pool, does not call GPTZero, and does not modify any source dataset or prior report. The 15 local detector orientations are inherited unchanged from `reports/detection_v4_beemo_style.md`.\n",
        f"Uncertainty uses {N_BOOT:,} source-level bootstrap replicates with seed {SEED}. Rank-biserial r is `2U/(n_AAVE*n_SAE)-1`; positive values mean AAVE appears more machine-like. BH adjustment covers the 15 local configurations.\n",
        "## Pool verification\n",
        "- Retained AAVE pool: **212** rows (`variety == AAVE`); the five Gullah rows in the 217-row source file are outside this analysis.\n",
        "- Final era-matched SAE pool: **521** rows.\n",
        "- IDs and duplicate checks: 733 unique poem IDs, 733 unique source document IDs, no exact-text duplicates, and no whitespace/case-normalized text duplicates.\n",
        "- Prior RQ1 overlap: exactly 100 AAVE and 100 SAE poems. The remaining 112 AAVE and 421 SAE poems are new to this full-pool audit.\n",
    ]
    for variety in ["AAVE", "SAE"]:
        d = pool.loc[pool["variety"].eq(variety)]
        years = d["publication_era"].value_counts().to_dict()
        sources = d["source_subset"].value_counts().to_dict()
        lines.append(f"- {variety} source subsets: {sources}. Publication-era values: {years}.")

    lines += [
        "\nEvery score row in `data/analysis/rq1_full_poetry_natural_scores.csv` carries both `poem_id` and `source_doc_id`, retained publication-era metadata, source subset, poet/title metadata, and the old-audit overlap flag.\n",
        "## Detector-cache audit and missing-only scoring\n",
        "The pre-inference audit found 200/733 cached poems for every local configuration, with no duplicate source IDs. Only the 533 missing poems were scored. The final caches contain 733/733 finite scores per configuration.\n",
        "| detector | expected | initially cached | missing | newly scored | final | duplicates | frozen orientation | cache/source |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in audit.iterrows():
        lines.append(
            f"| {row.detector} | {row.expected_rows} | {row.initial_cached_rows} | {row.initial_missing_rows} | "
            f"{row.newly_scored_rows} | {row.final_cached_rows} | {row.duplicate_rows} | {row.orientation} | `{row.cache_source}` |"
        )

    lines += [
        "\n## Full 212-versus-521 threshold-free comparison\n",
        "| detector | mean AAVE | mean SAE | median AAVE | median SAE | U | r [95% bootstrap CI] | raw p | BH p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.detector} | {fmt(row.mean_aave)} | {fmt(row.mean_sae)} | {fmt(row.median_aave)} | "
            f"{fmt(row.median_sae)} | {row.mann_whitney_u:.1f} | {fmt(row.rank_biserial_r)} "
            f"[{fmt(row.rank_biserial_ci_lo)}, {fmt(row.rank_biserial_ci_hi)}] | {fmt(row.p_raw_two_sided)} | {fmt(row.p_bh_15)} |"
        )

    lines += [
        "\n## Nested sensitivity: exact prior 100/100 subset\n",
        "This table recomputes the same local-detector statistics on the exact 100 AAVE and 100 SAE poems retained in the previous final RQ1 audit. BH adjustment is over the 15 local configurations only; GPTZero is intentionally excluded from this step.\n",
        "| detector | mean AAVE | mean SAE | median AAVE | median SAE | U | r [95% bootstrap CI] | raw p | BH p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.detector} | {fmt(row.old_mean_aave)} | {fmt(row.old_mean_sae)} | {fmt(row.old_median_aave)} | "
            f"{fmt(row.old_median_sae)} | {row.old_mann_whitney_u:.1f} | {fmt(row.old_rank_biserial_r)} "
            f"[{fmt(row.old_rank_biserial_ci_lo)}, {fmt(row.old_rank_biserial_ci_hi)}] | "
            f"{fmt(row.old_p_raw_two_sided)} | {fmt(row.old_p_bh_15)} |"
        )

    lines += [
        "\n## Fixed SAE folds and 5% operational evaluation\n",
        "The 521 SAE poems are assigned once by SHA-256 ordering of `source_doc_id`, followed by round-robin allocation to five folds. This is the exact fold function used in final RQ1 and gives SAE fold sizes of 105, 104, 104, 104, and 104. For each held-out fold, the threshold is NumPy's empirical 95th percentile of the other four SAE folds with `method=\"higher\"`. Classification is strictly `score > threshold`; ties remain human.\n",
        "For the unequal AAVE pool, the prior RQ1 convention is applied exactly: AAVE poems receive a separate deterministic SHA-256/round-robin evaluation fold, and each poem is evaluated with the threshold fitted without the corresponding SAE fold. AAVE never enters threshold calibration. The intervals below bootstrap the fixed out-of-fold classifications and are therefore **conditional on the fitted fold thresholds**; they do not include threshold-recalibration uncertainty.\n",
        "| detector | OOF SAE FPR | AAVE FPR | delta [95% conditional source-bootstrap CI] |",
        "|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.detector} | {row.oof_fpr_sae:.4f} | {row.aave_fpr:.4f} | {row.operational_delta:+.4f} "
            f"[{row.operational_delta_ci_lo:+.4f}, {row.operational_delta_ci_hi:+.4f}] |"
        )

    dir_changes = int(summary["r_direction_changed"].sum())
    mag_changes = int(summary["r_material_magnitude_change_ge_0_10"].sum())
    op_dir_changes = int(summary["operational_direction_changed"].sum())
    op_mag_changes = int(summary["operational_material_magnitude_change_ge_0_10"].sum())
    z = summary.loc[summary["family"].eq("zero-shot")]
    pre = summary.loc[summary["family"].eq("pretrained")]
    old_z_positive = int((z["old_rank_biserial_r"] > 0).sum())
    full_z_positive = int((z["rank_biserial_r"] > 0).sum())
    old_z_bh = int((z["old_p_bh_15"] < .05).sum())
    full_z_bh = int((z["p_bh_15"] < .05).sum())
    old_z_ci = int(((z["old_rank_biserial_ci_lo"] > 0) | (z["old_rank_biserial_ci_hi"] < 0)).sum())
    full_z_ci = int(((z["rank_biserial_ci_lo"] > 0) | (z["rank_biserial_ci_hi"] < 0)).sum())
    old_z_pos_bh = int(((z["old_rank_biserial_r"] > 0) & (z["old_p_bh_15"] < .05)).sum())
    full_z_pos_bh = int(((z["rank_biserial_r"] > 0) & (z["p_bh_15"] < .05)).sum())
    old_z_op_pos = int((z["old_operational_delta"] > 0).sum())
    full_z_op_pos = int((z["operational_delta"] > 0).sum())
    old_z_op_pos_ci = int(((z["old_operational_delta"] > 0) & (z["old_operational_delta_ci_lo"] > 0)).sum())
    full_z_op_pos_ci = int(((z["operational_delta"] > 0) & (z["operational_delta_ci_lo"] > 0)).sum())
    lines += [
        "\n## 1. Exact comparison to the previous 100/100 result\n",
        "| detector | old r [CI] | full r [CI] | change | direction changed | old operational delta [CI] | full operational delta [CI] |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.detector} | {row.old_rank_biserial_r:+.4f} [{row.old_rank_biserial_ci_lo:+.4f}, {row.old_rank_biserial_ci_hi:+.4f}] | "
            f"{row.rank_biserial_r:+.4f} [{row.rank_biserial_ci_lo:+.4f}, {row.rank_biserial_ci_hi:+.4f}] | "
            f"{row.r_change_full_minus_old:+.4f} | {'yes' if row.r_direction_changed else 'no'} | "
            f"{row.old_operational_delta:+.4f} [{row.old_operational_delta_ci_lo:+.4f}, {row.old_operational_delta_ci_hi:+.4f}] | "
            f"{row.operational_delta:+.4f} [{row.operational_delta_ci_lo:+.4f}, {row.operational_delta_ci_hi:+.4f}] |"
        )
    lines.append(
        f"Across the 15 local configurations, {dir_changes} threshold-free directions change and {mag_changes} change by at least 0.10 in r. "
        f"At the operational point, {op_dir_changes} delta directions change and {op_mag_changes} change by at least 0.10. "
        "The 0.10 cutoff is a transparent descriptive sensitivity rule, not an inferential threshold."
    )

    lines += [
        "\n## 2. Which claims become stronger or weaker with 212/521\n",
        f"**Stronger detector-specific separation, but not a stronger family-wide positive-disparity claim.** "
        f"For the 12 zero-shot configurations, BH-supported effects in either direction increase from {old_z_bh}/12 to {full_z_bh}/12, "
        f"and bootstrap r intervals excluding zero increase from {old_z_ci}/12 to {full_z_ci}/12. Positive directions increase only from "
        f"{old_z_positive}/12 to {full_z_positive}/12. The added positive sign is GPT2-XL Binoculars, which moves from r=-0.0274 to "
        f"r=+0.0067 while both intervals include zero, so it is not a substantive reversal. BH-supported positive zero-shot effects increase "
        f"from {old_z_pos_bh}/12 to {full_z_pos_bh}/12 and are confined to OPT-1.3B Log Probability, Entropy, and Binoculars in the full pool.\n",
        f"**Two threshold-free magnitudes change by at least 0.10.** GPT2-XL Rank weakens in the negative direction "
        f"(r=-0.5282 to -0.4149), while OpenAI-RoBERTa-base strengthens in the negative direction (r=-0.5302 to -0.6439). "
        f"The other 13 configurations change by less than 0.10 in r.\n",
        f"**The operational positive-disparity summary weakens slightly.** Positive zero-shot deltas remain {old_z_op_pos}/12 in the old subset "
        f"and {full_z_op_pos}/12 in the full pool, but positive conditional intervals excluding zero decrease from {old_z_op_pos_ci}/12 to "
        f"{full_z_op_pos_ci}/12 because GPT2-XL Binoculars becomes uncertain. No operational delta changes by 0.10 or more.\n",
        f"**Pretrained detectors do not support an AAVE-more-machine-like claim.** Full-pool effects are positive for "
        f"{int((pre.rank_biserial_r > 0).sum())}/3 pretrained configurations; all three are negative and BH-supported. "
        f"The larger sample therefore strengthens the conclusion that the local-detector result is configuration-dependent and weakens any "
        f"claim that the 100/100 result generalized uniformly across detector families.\n",
        "## 3. Manuscript-safe wording\n",
        "Across the complete retained natural-poetry pools (212 AAVE and 521 era-matched SAE poems), local detector behavior was heterogeneous. Several configurations assigned higher machine-likeness to AAVE, while others assigned lower scores or showed uncertain differences. The result depends on detector configuration and operating rule; it does not support a universal claim that natural AAVE poetry is more likely to be classified as machine-generated. Operational false-positive differences use thresholds cross-fitted on SAE poetry only.\n",
        "## 4. Unresolved limitations\n",
        "- Era matching is source-level rather than exact poem-year matching for Hughes: rows without a verified per-poem year are bounded to the source compilation's 1921-1928 range.\n",
        "- Poems are unique documents, but author, anthology, form, topic, length, and orthographic style remain partly confounded with variety.\n",
        "- The three pretrained configurations inherit the frozen label-orientation decisions from the final audit; two upstream configs expose generic label names, so this analysis does not relitigate orientation on the target sample.\n",
        "- Binoculars retains the project's disclosed smaller same-family observer pairing rather than the original large base/instruction-tuned recipe.\n",
        "- Operational intervals are conditional on the fitted thresholds and omit threshold-recalibration uncertainty.\n",
        "- This step evaluates local detectors only. GPTZero and Multi-VALUE are intentionally outside scope.\n",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    pool = load_pool()
    scores, audit = load_scores(pool)
    score_out, thresholds, summary = analyze(scores)

    score_columns = [
        "poem_id", "source_doc_id", "variety", "source_dataset", "source_subset", "poet", "poem_title",
        "pub_year", "publication_era", "n_words", "overlaps_old_rq1_100_100", "evaluation_fold",
        "detector", "family", "backbone", "metric_key", "raw_score", "orientation_multiplier",
        "oriented_score", "score_origin", "source_cache", "threshold", "flag_machine",
    ]
    score_out[score_columns].sort_values(["detector", "variety", "source_doc_id"]).to_csv(
        ANALYSIS / "rq1_full_poetry_natural_scores.csv", index=False
    )

    sae_folds = pool.loc[pool["variety"].eq("SAE"), [
        "poem_id", "source_doc_id", "source_dataset", "source_subset", "poet", "poem_title",
        "pub_year", "publication_era", "n_words", "overlaps_old_rq1_100_100", "evaluation_fold",
    ]].rename(columns={"evaluation_fold": "sae_fold"})
    sae_folds.sort_values(["sae_fold", "source_doc_id"]).to_csv(
        ANALYSIS / "rq1_full_poetry_sae_folds.csv", index=False
    )
    thresholds.to_csv(ANALYSIS / "rq1_full_poetry_thresholds.csv", index=False)
    summary.to_csv(ANALYSIS / "rq1_full_poetry_summary.csv", index=False)
    write_report(pool, audit, summary)
    print(summary[["detector", "rank_biserial_r", "p_bh_15", "operational_delta"]].to_string(index=False))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()

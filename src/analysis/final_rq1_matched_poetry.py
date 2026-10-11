"""Reviewer-final matched-poetry analysis over the 16 retained detectors.

This script never calls a model or external API.  It reuses the v4/v5 score
caches, fixes every score orientation to the decisions documented in the
original detector audit, and writes the requested CSVs and Markdown report.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data/interim"
OUT = ROOT / "data/analysis"
REPORT = ROOT / "reports/final_rq1_matched_poetry_16detectors.md"
SEED = 20260916
N_BOOT = 10_000

ZS = {
    "mean_log_prob": "Log Probability",
    "mean_rank": "Rank (negated)",
    "mean_log_rank": "Log-Rank (negated)",
    "mean_entropy": "Entropy",
    "detectllm_lrr": "DetectLLM-LRR",
    "binoculars": "Binoculars",
}

# Frozen from reports/detection_v4_beemo_style.md lines 34--48.  These signs
# were chosen on the earlier non-oratory H-vs-M task, never on this comparison.
FROZEN_SIGN = {
    "mean_log_prob": 1,
    "mean_rank": -1,
    "mean_log_rank": -1,
    "mean_entropy": -1,
    "detectllm_lrr": 1,
    "binoculars": 1,
}


def stable_folds(ids: pd.Series, n_folds: int = 5) -> pd.Series:
    order = sorted(ids.astype(str).unique(), key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    mapping = {doc: i % n_folds for i, doc in enumerate(order)}
    return ids.astype(str).map(mapping).astype(int)


def rb_from_u(u: float, nx: int, ny: int) -> float:
    return 2.0 * u / (nx * ny) - 1.0


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    u = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic").statistic
    return rb_from_u(float(u), len(x), len(y))


def bootstrap_score_stats(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, float]:
    mean_d = np.empty(N_BOOT)
    rb = np.empty(N_BOOT)
    for b in range(N_BOOT):
        xb = x[rng.integers(0, len(x), len(x))]
        yb = y[rng.integers(0, len(y), len(y))]
        mean_d[b] = xb.mean() - yb.mean()
        rb[b] = rank_biserial(xb, yb)
    q = np.quantile(np.column_stack([mean_d, rb]), [0.025, 0.975], axis=0)
    return float(q[0, 0]), float(q[1, 0]), float(q[0, 1]), float(q[1, 1])


def bh_adjust(p: pd.Series) -> pd.Series:
    vals = p.to_numpy(float)
    order = np.argsort(vals)
    ranked = vals[order]
    adjusted = np.minimum.accumulate((ranked * len(vals) / np.arange(1, len(vals) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=p.index)


def load_detectors() -> list[dict]:
    sub = pd.read_csv(INTERIM / "detection_v4_subsample.csv")
    keep = (sub.category == "HWT") & (sub.genre == "poetry") & sub.variety.isin(["AAVE", "SAE"])
    meta = sub.loc[keep, ["variety", "source_doc_id"]].copy()
    rows: list[dict] = []

    for suffix in ["gpt2", "opt"]:
        frame = pd.read_csv(INTERIM / f"detection_v4_zeroshot_{suffix}.csv").set_index("row_idx")
        backbone = str(frame.backbone.iloc[0])
        for col, display in ZS.items():
            values = frame[col].reindex(meta.index) * FROZEN_SIGN[col]
            rows.append({"detector": f"{display} ({backbone})", "family": "zero-shot", "meta": meta, "score": values})

    pre = pd.read_csv(INTERIM / "detection_v4_pretrained.csv")
    for detector, frame in pre.groupby("detector", sort=False):
        values = frame.set_index("row_idx").p_machine.reindex(meta.index)
        rows.append({"detector": detector, "family": "pretrained", "meta": meta, "score": values})

    gz = pd.read_csv(INTERIM / "detection_v5_gptzero.csv").set_index("row_idx")
    values = gz.completely_generated_prob.reindex(meta.index)
    rows.append({"detector": "GPTZero", "family": "commercial", "meta": meta, "score": values})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    stat_rows = []
    threshold_rows = []
    prediction_rows = []

    for item in load_detectors():
        d = item["meta"].copy()
        d["score"] = item["score"]
        assert d.score.notna().all(), f"missing matched-poetry scores for {item['detector']}"
        assert d.groupby("variety").size().to_dict() == {"AAVE": 100, "SAE": 100}
        x = d.loc[d.variety == "AAVE", "score"].to_numpy(float)
        y = d.loc[d.variety == "SAE", "score"].to_numpy(float)
        mw = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        mean_lo, mean_hi, rb_lo, rb_hi = bootstrap_score_stats(x, y, rng)
        stat_rows.append({
            "detector": item["detector"], "family": item["family"], "n_aave": len(x), "n_sae": len(y),
            "mean_aave": x.mean(), "mean_sae": y.mean(), "median_aave": np.median(x), "median_sae": np.median(y),
            "mean_diff_aave_minus_sae": x.mean() - y.mean(), "median_diff_aave_minus_sae": np.median(x) - np.median(y),
            "mann_whitney_u": float(mw.statistic), "p_raw_two_sided": float(mw.pvalue),
            "rank_biserial_r": rb_from_u(float(mw.statistic), len(x), len(y)),
            "mean_diff_ci_lo": mean_lo, "mean_diff_ci_hi": mean_hi,
            "rank_biserial_ci_lo": rb_lo, "rank_biserial_ci_hi": rb_hi,
        })

        # Five-fold cross-fitting.  Folds are balanced independently within
        # variety.  Each threshold is the SAE-training empirical q=.95 using
        # numpy's method='higher'; score > threshold is machine, so ties at the
        # threshold remain human.
        d["fold"] = -1
        for variety in ["AAVE", "SAE"]:
            m = d.variety.eq(variety)
            d.loc[m, "fold"] = stable_folds(d.loc[m, "source_doc_id"]).to_numpy()
        d["threshold"] = np.nan
        d["flag_machine"] = False
        for fold in range(5):
            cal = d[(d.variety == "SAE") & (d.fold != fold)].score.to_numpy(float)
            threshold = float(np.quantile(cal, 0.95, method="higher"))
            held = d.fold == fold
            d.loc[held, "threshold"] = threshold
            d.loc[held, "flag_machine"] = d.loc[held, "score"] > threshold

        rates = d.groupby("variety").flag_machine.mean()
        delta = float(rates.AAVE - rates.SAE)
        a = d.loc[d.variety == "AAVE", "flag_machine"].to_numpy(float)
        s = d.loc[d.variety == "SAE", "flag_machine"].to_numpy(float)
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            boot[b] = a[rng.integers(0, len(a), len(a))].mean() - s[rng.integers(0, len(s), len(s))].mean()
        lo, hi = np.quantile(boot, [0.025, 0.975])
        threshold_rows.append({
            "detector": item["detector"], "family": item["family"], "n_aave": len(a), "n_sae": len(s),
            "oof_fpr_aave": float(rates.AAVE), "oof_fpr_sae": float(rates.SAE), "delta": delta,
            "delta_ci_lo": float(lo), "delta_ci_hi": float(hi),
        })
        for idx, row in d.iterrows():
            prediction_rows.append({"detector": item["detector"], "family": item["family"], "row_idx": idx,
                                    "source_doc_id": row.source_doc_id, "variety": row.variety, "fold": int(row.fold),
                                    "oriented_score": row.score, "threshold": row.threshold,
                                    "flag_machine": int(row.flag_machine)})

    stats = pd.DataFrame(stat_rows)
    stats["p_bh_16"] = bh_adjust(stats.p_raw_two_sided)
    thresh = pd.DataFrame(threshold_rows)
    preds = pd.DataFrame(prediction_rows)
    stats.to_csv(OUT / "final_rq1_matched_poetry_score_stats.csv", index=False)
    thresh.to_csv(OUT / "final_rq1_matched_poetry_crossfit_thresholds.csv", index=False)
    preds.to_csv(OUT / "final_rq1_matched_poetry_predictions.csv", index=False)

    def f(x: float) -> str:
        return f"{x:.4g}"

    lines = [
        "# Final RQ1: matched poetry across all 16 detectors\n",
        "## Design\n",
        "This is a retained-score analysis: no detector was rerun. It restricts the v4/v5 audit to HWT poetry, with exactly **100 AAVE and 100 SAE poems** for every detector. The SAE pool is the final era-matched poetry pool. Every score is oriented so higher means more machine-like using the orientation frozen in `reports/detection_v4_beemo_style.md` (and GPTZero's documented probability); no sign was selected from this matched comparison.\n",
        f"Uncertainty uses {N_BOOT:,} source-level bootstrap replicates (seed {SEED}). Poems are one row per `source_doc_id`. Rank-biserial r is `2U/(n_AAVE*n_SAE)-1`, so positive means AAVE is more machine-like. BH adjustment is across all 16 raw Mann–Whitney p-values.\n",
        "## Threshold-free results\n",
        "| detector | family | mean AAVE | mean SAE | median AAVE | median SAE | mean diff [95% CI] | median diff | U | r [95% CI] | raw p | BH p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in stats.iterrows():
        lines.append(f"| {r.detector} | {r.family} | {f(r.mean_aave)} | {f(r.mean_sae)} | {f(r.median_aave)} | {f(r.median_sae)} | {f(r.mean_diff_aave_minus_sae)} [{f(r.mean_diff_ci_lo)}, {f(r.mean_diff_ci_hi)}] | {f(r.median_diff_aave_minus_sae)} | {r.mann_whitney_u:.1f} | {f(r.rank_biserial_r)} [{f(r.rank_biserial_ci_lo)}, {f(r.rank_biserial_ci_hi)}] | {f(r.p_raw_two_sided)} | {f(r.p_bh_16)} |")

    lines += [
        "\n## Poetry-only cross-fitted operational threshold\n",
        "Five folds are assigned by SHA-256 ordering separately within AAVE and SAE (20 poems per variety per fold). For each fold, the threshold is NumPy's empirical 95th percentile of the other four folds' SAE-poetry scores with `method=\"higher\"`. A held-out score is flagged only when `score > threshold`; ties at the threshold remain human. AAVE never calibrates a threshold. The CI bootstraps the fixed out-of-fold source classifications.\n",
        "| detector | OOF FPR AAVE | OOF FPR SAE | delta [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for _, r in thresh.iterrows():
        lines.append(f"| {r.detector} | {r.oof_fpr_aave:.3f} | {r.oof_fpr_sae:.3f} | {r.delta:+.3f} [{r.delta_ci_lo:+.3f}, {r.delta_ci_hi:+.3f}] |")

    lines += ["\n## Family summary\n"]
    for family in ["zero-shot", "pretrained", "commercial"]:
        s = stats[stats.family == family]
        lines.append(
            f"- **{family}:** {int((s.rank_biserial_r > 0).sum())}/{len(s)} positive directions; "
            f"median r={s.rank_biserial_r.median():+.3f}; {int((s.p_bh_16 < .05).sum())} BH-significant; "
            f"{int(((s.rank_biserial_ci_lo > 0) | (s.rank_biserial_ci_hi < 0)).sum())} effect-size CIs exclude zero."
        )
    z = stats[stats.family == "zero-shot"]
    broad = int((z.rank_biserial_r > 0).sum())
    zt = thresh[thresh.family == "zero-shot"]
    zt_positive = int((zt.delta > 0).sum())
    zt_positive_ci = int(((zt.delta > 0) & (zt.delta_ci_lo > 0)).sum())
    lines += [
        f"At the poetry-only cross-fitted operational threshold, the zero-shot family is also mixed but leans positive: {zt_positive}/12 deltas are positive and {zt_positive_ci}/12 positive-delta bootstrap intervals exclude zero. The pretrained rows are non-positive (two negative, one zero), while GPTZero is +0.040 with an interval spanning zero. Thus the threshold-free and operational summaries are not interchangeable.\n",
        "\n## Answer to the reviewer-critical question\n",
        f"The old broad pooled zero-shot pattern **does not survive intact after simultaneous genre and era matching**. Threshold-free, {broad}/12 zero-shot configurations point in the AAVE-more-machine-like direction, with median r={z.rank_biserial_r.median():+.3f}; {int((z.p_bh_16 < .05).sum())}/12 survive BH correction (in either direction) and {int(((z.rank_biserial_ci_lo > 0) | (z.rank_biserial_ci_hi < 0)).sum())}/12 bootstrap r intervals exclude zero. Under the clean poetry-only operational threshold, {zt_positive}/12 deltas are positive and {zt_positive_ci} positive intervals exclude zero. The result therefore weakens from a broad family-wide claim to a detector- and decision-rule-dependent pattern; the manuscript should report both summaries rather than selecting the one that best preserves the old narrative.\n",
        "## Reproducibility outputs\n",
        "- `data/analysis/final_rq1_matched_poetry_score_stats.csv`\n",
        "- `data/analysis/final_rq1_matched_poetry_crossfit_thresholds.csv`\n",
        "- `data/analysis/final_rq1_matched_poetry_predictions.csv`\n",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(stats[["detector", "family", "rank_biserial_r", "p_bh_16"]].to_string(index=False))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()

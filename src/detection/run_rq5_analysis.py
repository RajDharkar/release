"""
RQ5: false-negative-rate analysis for the direct-AAVE arm.

Reuses, VERBATIM, the 5%-SAE-FPR-calibrated thresholds and per-detector
orientation signs from the v4/v5/v6 sessions (reports/detection_v6_threshold_metrics.md
section (b); signs recomputed from the original v4 subsample + score caches
via build_v4_report.orient_score(), not re-derived by eye -- see
scratchpad note in this session). No new calibration is performed here --
this task scores NEW text (the direct-AAVE arm, its paired SAE-MGT, and
Arm D) against thresholds fixed by OLD text (the v4 SAE-human calibration
set), by design (the brief: "at the SAME threshold used for the FPR
analysis").

Reads data/interim/rq5_subsample.csv, rq5_zeroshot_gpt2.csv,
rq5_zeroshot_opt.csv, rq5_pretrained.csv. Writes nothing but stdout (and,
via a separate call, the final report file) -- read-only on every existing
dataset per the brief.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generate import harness as H

INTERIM = H.INTERIM
SUBSAMPLE_PATH = INTERIM / "rq5_subsample.csv"

# -- fixed thresholds, reports/detection_v6_threshold_metrics.md section (b) --
THRESHOLDS = {
    ("zeroshot", "GPT2-XL", "mean_log_prob"): -2.323926,
    ("zeroshot", "GPT2-XL", "mean_rank"): -31.608826,
    ("zeroshot", "GPT2-XL", "mean_log_rank"): -1.126304,
    ("zeroshot", "GPT2-XL", "mean_entropy"): -2.282129,
    ("zeroshot", "GPT2-XL", "detectllm_lrr"): 2.114260,
    ("zeroshot", "GPT2-XL", "binoculars"): -3.485642,
    ("zeroshot", "OPT-1.3B", "mean_log_prob"): -2.366211,
    ("zeroshot", "OPT-1.3B", "mean_rank"): -32.205202,
    ("zeroshot", "OPT-1.3B", "mean_log_rank"): -1.130830,
    ("zeroshot", "OPT-1.3B", "mean_entropy"): -2.321289,
    ("zeroshot", "OPT-1.3B", "detectllm_lrr"): 2.078888,
    ("zeroshot", "OPT-1.3B", "binoculars"): -3.673156,
    ("pretrained", None, "OpenAI-RoBERTa-base"): 0.403624,
    ("pretrained", None, "OpenAI-RoBERTa-large"): 0.999378,
    ("pretrained", None, "AIGC-MPU"): 0.004029,
    ("gptzero", None, "GPTZero"): 0.002086,
}

# -- orientation signs, recomputed from the original v4 subsample via
# build_v4_report.orient_score() on the H_all_vs_M reference task --
SIGNS = {
    ("GPT2-XL", "mean_log_prob"): 1, ("GPT2-XL", "mean_rank"): -1,
    ("GPT2-XL", "mean_log_rank"): -1, ("GPT2-XL", "mean_entropy"): -1,
    ("GPT2-XL", "detectllm_lrr"): 1, ("GPT2-XL", "binoculars"): 1,
    ("OPT-1.3B", "mean_log_prob"): 1, ("OPT-1.3B", "mean_rank"): -1,
    ("OPT-1.3B", "mean_log_rank"): -1, ("OPT-1.3B", "mean_entropy"): -1,
    ("OPT-1.3B", "detectllm_lrr"): 1, ("OPT-1.3B", "binoculars"): 1,
    ("pretrained", "OpenAI-RoBERTa-base"): 1, ("pretrained", "OpenAI-RoBERTa-large"): 1,
    ("pretrained", "AIGC-MPU"): 1,
    ("gptzero", "GPTZero"): 1,
}

ZS_METRICS = ["mean_log_prob", "mean_rank", "mean_log_rank", "mean_entropy", "detectllm_lrr", "binoculars"]
ZS_DISPLAY = {
    "mean_log_prob": "Log Probability", "mean_rank": "Rank (negated)",
    "mean_log_rank": "Log-Rank (negated)", "mean_entropy": "Entropy",
    "detectllm_lrr": "DetectLLM-LRR", "binoculars": "Binoculars",
}

# from reports/detection_v4_beemo_style.md's equity table (5%-SAE-FPR-matched
# threshold) -- the ALREADY-COMPUTED false-positive disparity, reused
# verbatim for STEP 6's direction comparison, not recomputed here (no new
# human text is scored by this script).
FP_SAE = {det: (0.050, 200) for det in [
    "Log Probability (GPT2-XL)", "Rank (negated) (GPT2-XL)", "Log-Rank (negated) (GPT2-XL)",
    "Entropy (GPT2-XL)", "DetectLLM-LRR (GPT2-XL)", "Binoculars (GPT2-XL)",
    "Log Probability (OPT-1.3B)", "Rank (negated) (OPT-1.3B)", "Log-Rank (negated) (OPT-1.3B)",
    "Entropy (OPT-1.3B)", "DetectLLM-LRR (OPT-1.3B)", "Binoculars (OPT-1.3B)",
    "OpenAI-RoBERTa-base", "OpenAI-RoBERTa-large", "AIGC-MPU", "GPTZero",
]}
FP_AAVE = {
    "Log Probability (GPT2-XL)": (0.093, 300), "Rank (negated) (GPT2-XL)": (0.060, 300),
    "Log-Rank (negated) (GPT2-XL)": (0.120, 300), "Entropy (GPT2-XL)": (0.143, 300),
    "DetectLLM-LRR (GPT2-XL)": (0.157, 300), "Binoculars (GPT2-XL)": (0.037, 300),
    "Log Probability (OPT-1.3B)": (0.227, 300), "Rank (negated) (OPT-1.3B)": (0.130, 300),
    "Log-Rank (negated) (OPT-1.3B)": (0.183, 300), "Entropy (OPT-1.3B)": (0.217, 300),
    "DetectLLM-LRR (OPT-1.3B)": (0.103, 300), "Binoculars (OPT-1.3B)": (0.213, 300),
    "OpenAI-RoBERTa-base": (0.073, 300), "OpenAI-RoBERTa-large": (0.073, 300), "AIGC-MPU": (0.013, 300),
    # reports/detection_v5_gptzero.md STEP 4(b): AAVE pooled FPR=0.137 (n=300,k=41) vs
    # SAE pooled FPR=0.050 (n=200,k=10) at the same 0.002086 threshold; dialect gap
    # +0.087, CIs do not overlap (statistically detectable) -- reused verbatim, not
    # recomputed (no new human text is scored by this script).
    "GPTZero": (0.137, 300),
}
FP_DISPARITY = {det: round(FP_AAVE[det][0] - FP_SAE[det][0], 3) for det in FP_SAE}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fnr(scores: pd.Series, threshold: float) -> tuple[float, int, int, tuple]:
    """FNR = fraction of MACHINE text scored <= threshold (missed, i.e.
    classified as human). k = number missed, n = total scored."""
    valid = scores.dropna()
    n = len(valid)
    k = int((valid <= threshold).sum())
    rate = k / n if n else float("nan")
    return rate, k, n, wilson_ci(k, n)


def load_all_scores() -> pd.DataFrame:
    """One row per (rq5_subsample row_idx, detector), oriented score, plus
    genre/category/variety/generator/source_doc_id/generation_id."""
    sub = pd.read_csv(SUBSAMPLE_PATH)
    frames = []

    gpt2_path = INTERIM / "rq5_zeroshot_gpt2.csv"
    opt_path = INTERIM / "rq5_zeroshot_opt.csv"
    pre_path = INTERIM / "rq5_pretrained.csv"

    for path, backbone in [(gpt2_path, "GPT2-XL"), (opt_path, "OPT-1.3B")]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for metric in ZS_METRICS:
            oriented = df[metric] * SIGNS[(backbone, metric)]
            frames.append(pd.DataFrame({
                "row_idx": df["row_idx"], "detector": f"{ZS_DISPLAY[metric]} ({backbone})",
                "family": "zeroshot", "score": oriented,
            }))

    if pre_path.exists():
        pre = pd.read_csv(pre_path)
        for det in pre.detector.unique():
            d = pre[pre.detector == det]
            oriented = d["p_machine"] * SIGNS[("pretrained", det)]
            frames.append(pd.DataFrame({
                "row_idx": d["row_idx"], "detector": det, "family": "pretrained", "score": oriented,
            }))

    gz_path = INTERIM / "rq5_gptzero.csv"
    if gz_path.exists():
        gz = pd.read_csv(gz_path)
        oriented = gz["completely_generated_prob"] * SIGNS[("gptzero", "GPTZero")]
        frames.append(pd.DataFrame({
            "row_idx": gz["row_idx"], "detector": "GPTZero", "family": "gptzero", "score": oriented,
        }))

    if not frames:
        return pd.DataFrame()
    scores = pd.concat(frames, ignore_index=True)
    merged = scores.merge(sub.reset_index().rename(columns={"index": "row_idx"}), on="row_idx", how="left")
    return merged


def threshold_for(detector: str) -> float:
    for (family, backbone, metric), thr in THRESHOLDS.items():
        if family == "zeroshot" and f"{ZS_DISPLAY[metric]} ({backbone})" == detector:
            return thr
        if family in ("pretrained", "gptzero") and metric == detector:
            return thr
    raise KeyError(detector)


def fnr_row(d: pd.DataFrame, category: str, thr: float, genre=None, generator=None) -> tuple[float, int, int, tuple]:
    sub = d[d.category == category]
    if genre is not None:
        sub = sub[sub.genre == genre]
    if generator is not None:
        sub = sub[sub.generator == generator]
    return fnr(sub.score, thr)


def step2_pooled(merged: pd.DataFrame, detectors: list[str]) -> dict:
    """Pooled FNR_SAE/FNR_AAVE(fiction+poetry)/FNR_lyrics/FNR_ArmD, D-. Returns
    dict[detector] -> results, for reuse by the disparity-direction step."""
    results = {}
    print("=" * 70)
    print("STEP 2 -- FNR pooled (fiction+poetry for SAE-paired comparison; lyrics separate)")
    print("=" * 70)
    for det in detectors:
        d = merged[merged.detector == det]
        thr = threshold_for(det)
        r_sae, k_sae, n_sae, ci_sae = fnr_row(d, "SAE_MGT", thr)
        aave_fp = d[(d.category == "AAVE_DIRECT") & (d.genre.isin(["fiction", "poetry"]))]
        r_aave, k_aave, n_aave, ci_aave = fnr(aave_fp.score, thr)
        aave_all = d[d.category == "AAVE_DIRECT"]
        r_aave_all, k_aave_all, n_aave_all, ci_aave_all = fnr(aave_all.score, thr)
        r_lyr, k_lyr, n_lyr, ci_lyr = fnr_row(d, "AAVE_DIRECT", thr, genre="lyrics")
        r_armd, k_armd, n_armd, ci_armd = fnr_row(d, "ARM_D", thr)
        d_minus = r_aave - r_sae if n_sae and n_aave else float("nan")

        results[det] = dict(r_sae=r_sae, n_sae=n_sae, k_sae=k_sae, ci_sae=ci_sae,
                             r_aave=r_aave, n_aave=n_aave, k_aave=k_aave, ci_aave=ci_aave,
                             r_aave_all=r_aave_all, n_aave_all=n_aave_all, ci_aave_all=ci_aave_all,
                             r_lyr=r_lyr, n_lyr=n_lyr, k_lyr=k_lyr, ci_lyr=ci_lyr,
                             r_armd=r_armd, n_armd=n_armd, k_armd=k_armd, ci_armd=ci_armd,
                             d_minus=d_minus, thr=thr)
        print(f"{det:32s} thr={thr:>10.4f}")
        print(f"    FNR_SAE(fic+poe)      ={r_sae:.3f} [{ci_sae[0]:.3f},{ci_sae[1]:.3f}] n={n_sae} k={k_sae}")
        print(f"    FNR_AAVE(fic+poe)     ={r_aave:.3f} [{ci_aave[0]:.3f},{ci_aave[1]:.3f}] n={n_aave} k={k_aave}  D-={d_minus:+.3f}")
        print(f"    FNR_AAVE(all 3 genre) ={r_aave_all:.3f} [{ci_aave_all[0]:.3f},{ci_aave_all[1]:.3f}] n={n_aave_all}")
        print(f"    FNR_lyrics (no SAE)   ={r_lyr:.3f} [{ci_lyr[0]:.3f},{ci_lyr[1]:.3f}] n={n_lyr} k={k_lyr}")
        print(f"    FNR_ArmD(fic+poe)     ={r_armd:.3f} [{ci_armd[0]:.3f},{ci_armd[1]:.3f}] n={n_armd} k={k_armd}")
    return results


def step2_per_genre_generator(merged: pd.DataFrame, detectors: list[str]):
    print("\n" + "=" * 70)
    print("STEP 2 -- FNR per genre")
    print("=" * 70)
    for det in detectors:
        d = merged[merged.detector == det]
        thr = threshold_for(det)
        print(f"\n{det} (thr={thr:.4f})")
        for genre in ["fiction", "poetry", "lyrics"]:
            r_sae, k_sae, n_sae, ci_sae = fnr_row(d, "SAE_MGT", thr, genre=genre)
            r_aave, k_aave, n_aave, ci_aave = fnr_row(d, "AAVE_DIRECT", thr, genre=genre)
            r_armd, k_armd, n_armd, ci_armd = fnr_row(d, "ARM_D", thr, genre=genre)
            dm = r_aave - r_sae if n_sae and n_aave else float("nan")
            dm_str = f"D-={dm:+.3f}" if n_sae else "D-=N/A (no SAE-lyrics cell)"
            print(f"  {genre:8s} FNR_SAE={r_sae:.3f}(n={n_sae},k={k_sae}) "
                  f"FNR_AAVE={r_aave:.3f}(n={n_aave},k={k_aave}) {dm_str} "
                  f"FNR_ArmD={r_armd:.3f}(n={n_armd},k={k_armd})")

    print("\n" + "=" * 70)
    print("STEP 2 -- FNR per generator (genres with an SAE_MGT cell: fiction+poetry)")
    print("=" * 70)
    for det in detectors:
        d = merged[merged.detector == det]
        thr = threshold_for(det)
        print(f"\n{det} (thr={thr:.4f})")
        for genre in ["fiction", "poetry"]:
            for gen in ["qwen3", "llama", "deepseek"]:
                r_sae, k_sae, n_sae, ci_sae = fnr_row(d, "SAE_MGT", thr, genre=genre, generator=gen)
                r_aave, k_aave, n_aave, ci_aave = fnr_row(d, "AAVE_DIRECT", thr, genre=genre, generator=gen)
                dm = r_aave - r_sae if n_sae and n_aave else float("nan")
                print(f"  {genre:8s} {gen:8s} FNR_SAE={r_sae:.3f}(n={n_sae}) "
                      f"FNR_AAVE={r_aave:.3f}(n={n_aave}) D-={dm:+.3f}" if n_sae and n_aave else
                      f"  {genre:8s} {gen:8s} FNR_SAE={r_sae:.3f}(n={n_sae}) FNR_AAVE={r_aave:.3f}(n={n_aave})")


def step3_paired(merged: pd.DataFrame, detectors: list[str]):
    print("\n" + "=" * 70)
    print("STEP 3 -- PAIRED within-document comparison (AAVE-direct vs its own SAE-MGT sibling)")
    print("=" * 70)
    for det in detectors:
        d = merged[merged.detector == det]
        aave = d[(d.category == "AAVE_DIRECT") & d.paired_mgt_generation_id.notna() & (d.paired_mgt_generation_id != "")]
        sae = d[d.category == "SAE_MGT"][["generation_id", "score", "genre"]].rename(
            columns={"generation_id": "paired_mgt_generation_id", "score": "sae_score"})
        paired = aave.merge(sae, on="paired_mgt_generation_id", suffixes=("", "_sae"))
        if paired.empty:
            print(f"{det}: no pairs found")
            continue
        diff = paired["score"] - paired["sae_score"]
        n = len(diff)
        n_more_machine = int((diff > 0).sum())
        n_less_machine = int((diff < 0).sum())
        n_tied = int((diff == 0).sum())
        prop = n_more_machine / n if n else float("nan")
        ci = wilson_ci(n_more_machine, n)
        try:
            stat, p = wilcoxon(diff) if (diff != 0).any() else (float("nan"), float("nan"))
        except ValueError:
            stat, p = float("nan"), float("nan")
        print(f"{det:32s} n_pairs={n} AAVE-more-machine-like={n_more_machine} ({prop:.3f} [{ci[0]:.3f},{ci[1]:.3f}]) "
              f"tied={n_tied} wilcoxon_p={p:.4g}" if p == p else
              f"{det:32s} n_pairs={n} AAVE-more-machine-like={n_more_machine} ({prop:.3f}) tied={n_tied}")

        for genre in ["fiction", "poetry"]:
            g = paired[paired.genre == genre]
            if g.empty:
                continue
            gd = g["score"] - g["sae_score"]
            gn = len(gd)
            gmore = int((gd > 0).sum())
            gci = wilson_ci(gmore, gn)
            print(f"    {genre:8s} n={gn} AAVE-more-machine-like={gmore} ({gmore/gn:.3f} [{gci[0]:.3f},{gci[1]:.3f}])")


def step5_two_by_two(merged: pd.DataFrame, detectors: list[str], pooled: dict):
    print("\n" + "=" * 70)
    print("STEP 5 -- 2x2: error type x variety (n per cell)")
    print("=" * 70)
    for det in detectors:
        res = pooled[det]
        print(f"\n{det}")
        if det in FP_SAE:
            fp_sae_rate, fp_sae_n = FP_SAE[det]
            fp_aave_rate, fp_aave_n = FP_AAVE[det]
            print(f"                  variety=AAVE                          variety=SAE")
            print(f"  FALSE POSITIVE  rate={fp_aave_rate:.3f} n={fp_aave_n} k={round(fp_aave_rate*fp_aave_n)}"
                  f"   (human text)      rate={fp_sae_rate:.3f} n={fp_sae_n} k={round(fp_sae_rate*fp_sae_n)}  (human text)")
        else:
            print(f"  FALSE POSITIVE  UNFILLABLE -- no comparable point in the existing v4 equity table for this "
                  f"detector (GPTZero was not one of that table's rows; see reports/detection_v5_gptzero.md)")
        print(f"  FALSE NEGATIVE  rate={res['r_aave_all']:.3f} n={res['n_aave_all']} k={round(res['r_aave_all']*res['n_aave_all'])}"
              f"  (direct-AAVE-MGT)  rate={res['r_sae']:.3f} n={res['n_sae']} k={res['k_sae']}  (SAE-MGT, fic+poe)")
        print(f"                  [Arm D, rule-perturbed, also variety=AAVE, separate mechanism: "
              f"FN rate={res['r_armd']:.3f} n={res['n_armd']} k={res['k_armd']}]")
        print(f"                  [lyrics has no SAE cell at all: FP|SAE and the SAE row above exclude lyrics "
              f"by construction; FN|AAVE above pools fiction+poetry+lyrics]")


def step6_direction(pooled: dict, detectors: list[str]):
    print("\n" + "=" * 70)
    print("STEP 6 -- does the miss disparity (D-) run the same direction as the FP disparity?")
    print("=" * 70)
    same = 0
    opposite = 0
    for det in detectors:
        d_minus = pooled[det]["d_minus"]
        fp_gap = FP_DISPARITY.get(det, float("nan"))
        if pd.isna(d_minus) or pd.isna(fp_gap):
            continue
        direction = "SAME" if (d_minus > 0) == (fp_gap > 0) else "OPPOSITE"
        if direction == "SAME":
            same += 1
        else:
            opposite += 1
        print(f"{det:32s} D-={d_minus:+.3f}  FP_gap={fp_gap:+.3f}  -> {direction}")
    print(f"\n{same}/{same+opposite} detectors: miss disparity runs the SAME direction as the FP disparity.")
    print(f"{opposite}/{same+opposite} detectors: OPPOSITE direction.")


def main():
    merged = load_all_scores()
    if merged.empty:
        print("No score files found yet.")
        return

    detectors = sorted(merged.detector.unique())
    print(f"Detectors with scores available: {len(detectors)} / 15")
    print(detectors)
    print()

    pooled = step2_pooled(merged, detectors)
    step2_per_genre_generator(merged, detectors)
    step3_paired(merged, detectors)
    step5_two_by_two(merged, detectors, pooled)
    step6_direction(pooled, detectors)


if __name__ == "__main__":
    main()

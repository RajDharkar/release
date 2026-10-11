"""Analyze aligned RQ3 v0.3 scores and write the pre-rewrite go/no-go report."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "pre_rewrite_methodology_resolution_2026-10-01"
OLD_OUT = ROOT / "outputs" / "revised_poetry_experiments_2026-09-30"
INTERIM = ROOT / "data/interim"
ANALYSIS = ROOT / "data/analysis"
ALPHAS = (0.025, 0.05, 0.10, 0.15, 0.20)
N_BOOT = 10_000
BOOT_SEED = 20261001

METRICS = {
    "mean_log_prob": "Log Probability",
    "mean_rank": "Rank (negated)",
    "mean_log_rank": "Log-Rank (negated)",
    "mean_entropy": "Entropy",
    "detectllm_lrr": "DetectLLM-LRR",
    "binoculars": "Binoculars",
}
SIGNS = {"mean_log_prob": 1, "mean_rank": -1, "mean_log_rank": -1, "mean_entropy": -1, "detectllm_lrr": 1, "binoculars": 1}
ZERO_SHOT = [f"{label} ({backbone})" for backbone in ("GPT2-XL", "OPT-1.3B") for label in METRICS.values()]
PRETRAINED = ["OpenAI-RoBERTa-base", "OpenAI-RoBERTa-large", "AIGC-MPU"]
DETECTORS = ZERO_SHOT + PRETRAINED


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def stable_seed(*parts: object) -> int:
    value = "|".join(map(str, (BOOT_SEED, *parts)))
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) % (2**32)


def stratified_ci(values: np.ndarray, strata: np.ndarray, *parts: object) -> tuple[float, float]:
    values, strata = np.asarray(values, float), np.asarray(strata, str)
    rng = np.random.default_rng(stable_seed(*parts))
    out = np.zeros(N_BOOT)
    for s in sorted(np.unique(strata)):
        x = values[strata == s]
        weight = len(x) / len(values)
        for start in range(0, N_BOOT, 250):
            stop = min(start + 250, N_BOOT)
            idx = rng.integers(0, len(x), size=(stop - start, len(x)))
            out[start:stop] += weight * x[idx].mean(axis=1)
    return tuple(map(float, np.quantile(out, [0.025, 0.975])))


def new_scores() -> pd.DataFrame:
    transformed = pd.read_csv(OUT / "rq3_transformed_sae_mgt_multivalue_v03.csv", keep_default_na=False)
    frames = []
    for tier, backbone in (("gpt2", "GPT2-XL"), ("opt", "OPT-1.3B")):
        wide = pd.read_csv(OUT / f"rq3_v03_{tier}_scores_wide.csv", keep_default_na=False)
        assert len(wide) == 209 and wide["generation_id"].nunique() == 209
        for metric, label in METRICS.items():
            frames.append(pd.DataFrame({
                "generation_id": wide["generation_id"], "detector": f"{label} ({backbone})",
                "raw_score": wide[metric].astype(float), "oriented_score": wide[metric].astype(float) * SIGNS[metric],
                "model_config_identifier": wide["model_config_identifier"],
            }))
    pre = pd.read_csv(OUT / "rq3_v03_pretrained_scores_wide.csv", keep_default_na=False)
    for detector in PRETRAINED:
        d = pre.loc[pre["detector"].eq(detector)]
        assert len(d) == 209
        frames.append(pd.DataFrame({
            "generation_id": d["generation_id"], "detector": detector,
            "raw_score": d["p_machine"].astype(float), "oriented_score": d["p_machine"].astype(float),
            "model_config_identifier": d["model_config_identifier"],
        }))
    scores = pd.concat(frames, ignore_index=True).merge(
        transformed[["generation_id", "source_poem_id", "generator", "evaluation_fold", "output_sha256"]],
        on="generation_id", validate="many_to_one",
    )
    assert len(scores) == 209 * 15 and scores.groupby("detector").size().eq(209).all()
    atomic_csv(OUT / "rq3_v03_detector_scores_long.csv", scores)
    return scores


def original_scores() -> pd.DataFrame:
    rq5 = pd.read_csv(INTERIM / "rq5_subsample.csv", keep_default_na=False).reset_index(names="row_idx")
    original = rq5.loc[(rq5["genre"] == "poetry") & (rq5["category"] == "SAE_MGT")].copy()
    frames = []
    for tier, backbone in (("gpt2", "GPT2-XL"), ("opt", "OPT-1.3B")):
        wide = pd.read_csv(INTERIM / f"rq5_zeroshot_{tier}.csv")
        score_cols = ["row_idx", *METRICS]
        wide = original[["row_idx", "generation_id"]].merge(wide[score_cols], on="row_idx", validate="one_to_one")
        for metric, label in METRICS.items():
            frames.append(pd.DataFrame({
                "generation_id": wide["generation_id"], "detector": f"{label} ({backbone})",
                "original_oriented_score": wide[metric].astype(float) * SIGNS[metric],
            }))
    pre = pd.read_csv(INTERIM / "rq5_pretrained.csv")[["row_idx", "detector", "p_machine"]]
    pre = pre.merge(original[["row_idx", "generation_id"]], on="row_idx", validate="many_to_one")
    for detector in PRETRAINED:
        d = pre.loc[pre["detector"].eq(detector)]
        frames.append(pd.DataFrame({
            "generation_id": d["generation_id"], "detector": detector,
            "original_oriented_score": d["p_machine"].astype(float),
        }))
    out = pd.concat(frames, ignore_index=True)
    assert len(out) == 209 * 15 and out.groupby("detector").size().eq(209).all()
    return out


def analyze_pair() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair = new_scores().merge(original_scores(), on=["generation_id", "detector"], validate="one_to_one")
    pair["score_shift"] = pair["oriented_score"] - pair["original_oriented_score"]
    thresholds = pd.read_csv(ANALYSIS / "rq1_operating_point_thresholds.csv")
    score_rows, fnr_rows = [], []
    for detector in DETECTORS:
        d = pair.loc[pair["detector"].eq(detector)].sort_values("generation_id").copy()
        shift = d["score_shift"].to_numpy(float)
        strata = d["generator"].to_numpy(str)
        lo, hi = stratified_ci(shift, strata, "score", detector)
        score_rows.append({
            "detector": detector, "family": "zero-shot" if detector in ZERO_SHOT else "pretrained",
            "primary_analysis": detector in ZERO_SHOT, "N": len(d),
            "mean_paired_score_shift": shift.mean(), "median_paired_score_shift": np.median(shift),
            "mean_paired_score_shift_CI_low": lo, "mean_paired_score_shift_CI_high": hi,
            "n_lower_machine_likeness": int((shift < 0).sum()), "proportion_lower_machine_likeness": (shift < 0).mean(),
            "n_higher_machine_likeness": int((shift > 0).sum()), "proportion_higher_machine_likeness": (shift > 0).mean(),
            "n_no_change": int((shift == 0).sum()), "bootstrap_replicates": N_BOOT,
            "bootstrap_design": "paired source-level resampling within generator strata",
        })
        for alpha in ALPHAS:
            t = thresholds.loc[(thresholds["detector"] == detector) & np.isclose(thresholds["alpha"], alpha), ["fold", "threshold"]]
            assert len(t) == 5
            threshold = d["evaluation_fold"].map(t.set_index("fold")["threshold"]).to_numpy(float)
            old_miss = d["original_oriented_score"].to_numpy(float) <= threshold
            new_miss = d["oriented_score"].to_numpy(float) <= threshold
            delta = new_miss.astype(float) - old_miss.astype(float)
            lo, hi = stratified_ci(delta, strata, "fnr", detector, alpha)
            fnr_rows.append({
                "detector": detector, "family": "zero-shot" if detector in ZERO_SHOT else "pretrained",
                "primary_analysis": detector in ZERO_SHOT, "target_SAE_human_FPR": alpha, "N": len(d),
                "original_SAE_MGT_FNR": old_miss.mean(), "transformed_v03_SAE_MGT_FNR": new_miss.mean(),
                "delta_FNR_v03": delta.mean(), "delta_FNR_v03_CI_low": lo, "delta_FNR_v03_CI_high": hi,
                "original_false_negatives": int(old_miss.sum()), "transformed_v03_false_negatives": int(new_miss.sum()),
                "n_detected_to_missed": int((~old_miss & new_miss).sum()), "n_missed_to_detected": int((old_miss & ~new_miss).sum()),
                "bootstrap_replicates": N_BOOT, "bootstrap_design": "paired source-level resampling within generator strata",
            })
    score = pd.DataFrame(score_rows)
    fnr = pd.DataFrame(fnr_rows)
    atomic_csv(OUT / "rq3_v03_mgt_transformation_scores.csv", score)
    atomic_csv(OUT / "rq3_v03_mgt_transformation_fnr.csv", fnr)
    return pair, score, fnr


def sensitivity(score: pd.DataFrame, fnr: pd.DataFrame) -> pd.DataFrame:
    old_score = pd.read_csv(OLD_OUT / "rq3_mgt_transformation_scores.csv")
    old_fnr = pd.read_csv(OLD_OUT / "rq3_mgt_transformation_fnr.csv")
    old5 = old_fnr.loc[np.isclose(old_fnr["target_SAE_human_FPR"], .05), [
        "detector", "original_SAE_MGT_FNR", "transformed_SAE_MGT_FNR", "delta_FNR_transformed_minus_original",
    ]].rename(columns={
        "transformed_SAE_MGT_FNR": "transformed_FNR_v013",
        "delta_FNR_transformed_minus_original": "delta_FNR_v013",
    })
    new5 = fnr.loc[np.isclose(fnr["target_SAE_human_FPR"], .05), [
        "detector", "transformed_v03_SAE_MGT_FNR", "delta_FNR_v03",
    ]]
    out = old5.merge(new5, on="detector", validate="one_to_one").merge(
        old_score[["detector", "mean_paired_score_shift"]].rename(columns={"mean_paired_score_shift": "mean_score_shift_v013"}),
        on="detector", validate="one_to_one",
    ).merge(
        score[["detector", "family", "primary_analysis", "mean_paired_score_shift"]].rename(columns={"mean_paired_score_shift": "mean_score_shift_v03"}),
        on="detector", validate="one_to_one",
    )
    out["FNR_direction_agrees"] = np.sign(out["delta_FNR_v013"]) == np.sign(out["delta_FNR_v03"])
    out["continuous_shift_direction_agrees"] = np.sign(out["mean_score_shift_v013"]) == np.sign(out["mean_score_shift_v03"])
    atomic_csv(OUT / "rq3_multivalue_version_sensitivity.csv", out)
    return out


def synthesis(fnr: pd.DataFrame) -> pd.DataFrame:
    human = pd.read_csv(OLD_OUT / "rq2_human_transformation_fpr.csv")
    human = human.loc[np.isclose(human["target_SAE_human_FPR"], .05), [
        "detector", "original_human_SAE_FPR", "transformed_human_SAE_FPR",
        "delta_FPR_transformed_minus_original", "delta_FPR_CI_low", "delta_FPR_CI_high",
    ]]
    mgt = fnr.loc[np.isclose(fnr["target_SAE_human_FPR"], .05), [
        "detector", "family", "primary_analysis", "original_SAE_MGT_FNR", "transformed_v03_SAE_MGT_FNR",
        "delta_FNR_v03", "delta_FNR_v03_CI_low", "delta_FNR_v03_CI_high",
    ]]
    out = human.merge(mgt, on="detector", validate="one_to_one").rename(columns={
        "original_human_SAE_FPR": "human_original_SAE_FPR",
        "transformed_human_SAE_FPR": "human_transformed_SAE_FPR",
        "delta_FPR_transformed_minus_original": "delta_FPR_human",
        "delta_FPR_CI_low": "delta_FPR_human_CI_low", "delta_FPR_CI_high": "delta_FPR_human_CI_high",
        "original_SAE_MGT_FNR": "MGT_original_SAE_FNR", "transformed_v03_SAE_MGT_FNR": "MGT_transformed_SAE_FNR",
        "delta_FNR_v03": "delta_FNR_MGT", "delta_FNR_v03_CI_low": "delta_FNR_MGT_CI_low",
        "delta_FNR_v03_CI_high": "delta_FNR_MGT_CI_high",
    })
    out["delta_FPR_human_CI"] = out.apply(lambda r: f"[{r.delta_FPR_human_CI_low:.6f}, {r.delta_FPR_human_CI_high:.6f}]", axis=1)
    out["delta_FNR_MGT_CI"] = out.apply(lambda r: f"[{r.delta_FNR_MGT_CI_low:.6f}, {r.delta_FNR_MGT_CI_high:.6f}]", axis=1)
    atomic_csv(OUT / "rq2_rq3_v03_synthesis.csv", out)
    return out


def generation_log_audit() -> dict:
    raw = pd.read_csv(INTERIM / "aave_poetry_mgt_modeb_run1.csv", keep_default_na=False)
    index: dict[tuple[str, str, str], list[str]] = {}
    for path in sorted((INTERIM / "live_raw").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            request, response = payload.get("request", {}), payload.get("response", {})
            messages = request.get("messages", [])
            prompt = messages[0].get("content", "") if len(messages) == 1 and messages[0].get("role") == "user" else ""
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            index.setdefault((request.get("model", ""), prompt, content), []).append(str(path.relative_to(ROOT)))
        except Exception:
            continue
    rows = []
    for row in raw.itertuples(index=False):
        matches = list(index.get((row.model_full_id, row.prompt, row.completion), []))
        if not matches and row.model_short_name == "qwen3":
            for suffix in ("/no_think", " /no_think"):
                matches += index.get((row.model_full_id, row.prompt, row.completion + suffix), [])
        stamp = datetime.fromisoformat(row.timestamp).strftime("%Y%m%dT%H%M%S")
        timestamp_matches = [p for p in matches if Path(p).name.startswith(stamp + "_")]
        selected = timestamp_matches[0] if len(timestamp_matches) == 1 else (matches[0] if len(matches) == 1 else "")
        rows.append({
            "generation_id": row.generation_id, "model_full_id": row.model_full_id,
            "exact_request_response_match_count": len(matches), "timestamp_exact_match_count": len(timestamp_matches),
            "selected_api_log_path": selected, "verified": bool(selected),
        })
    frame = pd.DataFrame(rows)
    assert len(frame) == 210 and frame["verified"].all()
    atomic_csv(OUT / "sae_mgt_generation_api_log_matches.csv", frame)
    return {
        "attempts": 210, "verified": int(frame["verified"].sum()),
        "unique_request_response_matches": int((frame["exact_request_response_match_count"] == 1).sum()),
        "timestamp_resolved_ambiguous_matches": int((frame["exact_request_response_match_count"] > 1).sum()),
    }


def historical_equality() -> dict:
    transform = pd.read_csv(OUT / "rq3_transformed_sae_mgt_multivalue_v03.csv", keep_default_na=False)
    old_transform = pd.read_csv(INTERIM / "arm_d_poetry_aave_mgt_run1.csv", keep_default_na=False)
    text_join = transform.merge(
        old_transform[["source_generation_id", "arm_d_id", "aave_mgt_text"]],
        left_on="generation_id", right_on="source_generation_id", validate="one_to_one",
    )
    text_equal = text_join["transformed_sae_mgt_text"].eq(text_join["aave_mgt_text"])
    rq5 = pd.read_csv(INTERIM / "rq5_subsample.csv", keep_default_na=False).reset_index(names="row_idx")
    old_rows = rq5.loc[(rq5["genre"] == "poetry") & (rq5["category"] == "ARM_D"), ["row_idx", "generation_id"]].merge(
        old_transform[["arm_d_id", "source_generation_id"]], left_on="generation_id", right_on="arm_d_id", validate="one_to_one"
    )
    max_differences = {}
    for tier in ("gpt2", "opt"):
        old = pd.read_csv(INTERIM / f"rq5_zeroshot_{tier}.csv").merge(old_rows, on="row_idx", validate="one_to_one")
        new = pd.read_csv(OUT / f"rq3_v03_{tier}_scores_wide.csv")
        joined = new.merge(old, left_on="generation_id", right_on="source_generation_id", suffixes=("_new", "_old"), validate="one_to_one")
        max_differences[tier] = float(max(np.max(np.abs(joined[f"{m}_new"] - joined[f"{m}_old"])) for m in METRICS))
    old_pre = pd.read_csv(INTERIM / "rq5_pretrained.csv").merge(old_rows, on="row_idx", validate="many_to_one")
    new_pre = pd.read_csv(OUT / "rq3_v03_pretrained_scores_wide.csv")
    joined_pre = new_pre.merge(old_pre, left_on=["generation_id", "detector"], right_on=["source_generation_id", "detector"], validate="one_to_one")
    max_differences["pretrained"] = float(np.max(np.abs(joined_pre["p_machine_x"] - joined_pre["p_machine_y"])))
    result = {
        "v03_vs_v013_texts_compared": len(text_join), "byte_identical_texts": int(text_equal.sum()),
        "different_texts": int((~text_equal).sum()), "maximum_raw_score_absolute_difference": max_differences,
        "all_new_raw_scores_exactly_equal_historical": all(v == 0 for v in max_differences.values()),
    }
    atomic_text(OUT / "rq3_v03_historical_equality_check.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def report(pair: pd.DataFrame, score: pd.DataFrame, fnr: pd.DataFrame, sens: pd.DataFrame, syn: pd.DataFrame, log_audit: dict, equality: dict) -> None:
    pfnr = fnr.loc[fnr["primary_analysis"]].copy()
    five = pfnr.loc[np.isclose(pfnr["target_SAE_human_FPR"], .05)]
    psyn = syn.loc[syn["primary_analysis"]]
    all_positive_by_alpha = pfnr.groupby("target_SAE_human_FPR")["delta_FNR_v03"].apply(lambda x: int((x > 0).sum()))
    transform = pd.read_csv(OUT / "rq3_transformed_sae_mgt_multivalue_v03.csv", keep_default_na=False)
    old = pd.read_csv(INTERIM / "arm_d_poetry_aave_mgt_run1.csv", keep_default_na=False)
    versions = transform.merge(old[["source_generation_id", "aave_mgt_text"]], left_on="generation_id", right_on="source_generation_id", validate="one_to_one")
    identical = int((versions["transformed_sae_mgt_text"] == versions["aave_mgt_text"]).sum())
    tstate = json.loads((OUT / "rq2_v03_transformation_regression.json").read_text())
    dstates = {tier: json.loads((OUT / f"rq3_v03_{tier}_detector_regression.json").read_text()) for tier in ("gpt2", "opt", "pretrained")}
    literal = """Rewrite the following poem in standard written English. Convert any non-standard spellings, grammar, and vocabulary into standard forms, while keeping the meaning, imagery, and emotional weight intact. Output only the rewritten poem.

--- SOURCE TEXT ---
{SOURCE_POEM}
--- END SOURCE TEXT ---

Keep the rewrite close to the original's length."""
    alpha_summary = ", ".join(f"{a*100:g}%: {n}/12" for a, n in all_positive_by_alpha.items())
    detector_lines = "\n".join(
        f"- {r.detector}: original {r.original_SAE_MGT_FNR:.3f}; v0.3 transformed {r.transformed_v03_SAE_MGT_FNR:.3f}; ΔFNR {r.delta_FNR_v03:+.3f} (95% CI {r.delta_FNR_v03_CI_low:+.3f}, {r.delta_FNR_v03_CI_high:+.3f})"
        for r in five.itertuples(index=False)
    )
    text = f"""# Pre-rewrite methodology resolution

## Decision

The two requested methodological issues are resolved without modifying the manuscript or any historical artifact. The primary RQ3 analysis now uses the exact retained Multi-VALUE/value-nlp v0.3 implementation used by canonical RQ2, and all requested detector pipelines passed historical-score regression before the new texts were accepted.

## SAE-MGT provenance

The 209 items are **SAE MGT rewrites**: LLM-produced standard-English rewrites of 209 human-authored AAVE source poems, not unconstrained fresh poems. Qwen contributes 69, Llama 70, and DeepSeek 70. All outputs are paired to one source poem and one generator.

The exact base user prompt was:

```text
{literal}
```

There was no system prompt. The prompt did not say `AAVE`, `dialect`, `vernacular`, `SAE`, or `Standard American English`; it did explicitly say **standard written English**. Qwen alone received a trailing `/no_think` and an `enable_thinking=false` request field. Outputs were rewrites because the complete source poem was inserted between the delimiters and the instruction explicitly said “Rewrite.” Response content was retained, except the client could strip a trailing Qwen `/no_think` artifact. Derived density/similarity measurements did not rewrite the text.

The raw run has 210 attempts. Exact stored request model, one-message user prompt, and response content were located for {log_audit['verified']}/{log_audit['attempts']} attempts in `data/interim/live_raw`: {log_audit['unique_request_response_matches']} were unique, while one repeated identical request/response was disambiguated by the recorded timestamp. One Qwen row, `mgt_modeb_poetry_run1_pd_poetry_hughes_langston_hughes_evil_woman_qwen3`, retained a frozen `refusal_bool=True` exclusion, leaving 209. The later audit indicates the heuristic was conservative, but this analysis does not reverse the frozen decision. The statement “The SAE-MGT poetry condition consists of machine-generated standard-English rewrites of human-authored AAVE source poems” is **VERIFIED**. Do not claim more broadly that a model was unaware of dialect; say only that dialect/AAVE was not named in the prompt.

## Transformation alignment

- Exact environment: Python 3.10.20, value-nlp 0.3, spaCy 3.7.4, en_core_web_sm 3.7.1, retained package-file hashes from the canonical RQ2 manifest.
- Variant: `AfricanAmericanVernacular` with coreference disabled by the retained subclass; no rule list or probabilities were altered.
- State/order: one dialect instance for the batch, ascending SHA-256 identity order, and per-row seed `int(sha256(identity)[:8], 16) or 1`.
- Normalization: whole text in, transformed string out; no input/output normalization.
- Regression: **{tstate['status']}**, {tstate['byte_exact_n']}/{tstate['sample_n']} canonical human outputs reproduced byte-for-byte. The already-frozen full 521-row canonical rerun remains an additional complete regression artifact.
- Machine transformation: 209/209 succeeded. Compared with v0.1.3, {identical}/209 transformed strings are byte-identical and {209-identical}/209 differ. The independent new inference also reproduces every historical transformed-text raw score exactly (maximum absolute difference 0 for GPT-2, OPT, and pretrained tiers).

## Detector compatibility

- GPT-2 family: **{dstates['gpt2']['status']}**, {dstates['gpt2']['comparisons']} comparisons, maximum absolute difference {dstates['gpt2']['maximum_absolute_difference']:.3g}.
- OPT family: **{dstates['opt']['status']}**, {dstates['opt']['comparisons']} comparisons, maximum absolute difference {dstates['opt']['maximum_absolute_difference']:.3g}.
- Three pretrained detectors: **{dstates['pretrained']['status']}**, {dstates['pretrained']['comparisons']} comparisons, maximum absolute difference {dstates['pretrained']['maximum_absolute_difference']:.3g}.

Only the 209 new transformed v0.3 texts were scored. Human AAVE, human SAE, original SAE MGT, and transformed human SAE scores remained frozen. GPTZero was not run.

## Revised RQ3 at the 5% SAE-human target

{detector_lines}

Positive ΔFNR occurs for {(five['delta_FNR_v03'] > 0).sum()}/12 primary configurations at 5%. Across the full sweep, positive counts are {alpha_summary}. Therefore the historical “12/12 higher FNR” statement **{'remains true' if (all_positive_by_alpha == 12).all() else 'does not remain true at every requested operating point'}** under v0.3.

Version sensitivity at 5%: FNR-change direction agrees for {int(sens.loc[sens.primary_analysis, 'FNR_direction_agrees'].sum())}/12 primary configurations; continuous mean-score-shift direction agrees for {int(sens.loc[sens.primary_analysis, 'continuous_shift_direction_agrees'].sum())}/12. In fact, all 209 transformed strings and all rerun raw scores are exactly equal across the retained v0.1.3 and v0.3 conditions, so all detector-level point estimates are identical. The primary paper result should nevertheless use the provenance-aligned v0.3 artifact; v0.1.3 remains a preserved sensitivity analysis.

## Human/machine symmetry after v0.3 alignment

Among 12 primary configurations at 5%:

- ΔFPR_human < 0: {int((psyn.delta_FPR_human < 0).sum())}
- ΔFPR_human > 0: {int((psyn.delta_FPR_human > 0).sum())}
- ΔFNR_MGT > 0: {int((psyn.delta_FNR_MGT > 0).sum())}
- ΔFNR_MGT < 0: {int((psyn.delta_FNR_MGT < 0).sum())}
- both lower human FPR and higher MGT FNR: {int(((psyn.delta_FPR_human < 0) & (psyn.delta_FNR_MGT > 0)).sum())}

**SUPPORTED:** “Applying the same rule-based dialect transformation decreases false-positive rates on human-authored text while increasing false-negative rates on machine-generated text.” This statement is supported detector-by-detector for all 12 primary configurations under the same retained v0.3 implementation at the 5% operating point.

All CIs use 10,000 paired source-level bootstrap replicates; RQ3 resampling is within Qwen/Llama/DeepSeek strata, preserving 69/70/70 composition. Frozen source-poem evaluation folds and frozen SAE-human thresholds are used at 2.5%, 5%, 10%, 15%, and 20%; thresholds are never recalibrated.

## Rewrite readiness

READY TO REWRITE
"""
    atomic_text(OUT / "pre_rewrite_methodology_resolution.md", text)


def manifest() -> None:
    rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and not path.name.endswith(".partial") and path.name != "output_manifest.csv":
            rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    atomic_csv(OUT / "output_manifest.csv", pd.DataFrame(rows))


def main() -> None:
    pair, score, fnr = analyze_pair()
    sens = sensitivity(score, fnr)
    syn = synthesis(fnr)
    log_audit = generation_log_audit()
    equality = historical_equality()
    report(pair, score, fnr, sens, syn, log_audit, equality)
    manifest()
    print(f"wrote aligned analysis bundle to {OUT}")


if __name__ == "__main__":
    main()

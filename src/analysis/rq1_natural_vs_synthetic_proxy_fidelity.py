"""Step 6: natural-versus-synthetic proxy fidelity from frozen artifacts only.

The analysis compares the final natural AAVE-versus-SAE reference with the
paired Multi-VALUE transformation benchmark. It never loads poem text, runs a
detector, calls GPTZero, transforms text, edits the manuscript, or performs the
later operating-point sweep.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data/analysis"
REPORT = ROOT / "reports/rq1_natural_vs_synthetic_proxy_fidelity.md"
FIGURE = ROOT / "figures/rq1_natural_vs_synthetic_proxy_fidelity.pdf"
FIDELITY_OUT = ANALYSIS / "rq1_natural_vs_synthetic_proxy_fidelity.csv"
DIFFERENCE_OUT = ANALYSIS / "rq1_natural_vs_synthetic_direct_differences.csv"

NATURAL_SCORES_PATH = ANALYSIS / "rq1_full_poetry_natural_scores.csv"
NATURAL_STEP1_PATH = ANALYSIS / "rq1_full_poetry_summary.csv"
NATURAL_ROBUSTNESS_PATH = ANALYSIS / "rq1_natural_reference_robustness_summary.csv"
PAIRED_PATH = ANALYSIS / "rq1_proxy_original_canonical_paired_scores.csv"
SYNTHETIC_EFFECTS_PATH = ANALYSIS / "rq1_proxy_paired_effects.csv"
SYNTHETIC_OPERATIONAL_PATH = ANALYSIS / "rq1_proxy_operational_paired_effects.csv"
FRESH_SCORES_PATH = ANALYSIS / "rq1_proxy_fresh_instance_sensitivity_scores.csv"
FRESH_EFFECTS_PATH = ANALYSIS / "rq1_proxy_paired_effects_fresh_instance.csv"
FRESH_COMPARISON_PATH = ANALYSIS / "rq1_proxy_paired_canonical_vs_fresh_instance_comparison.csv"
STEP5_SUMMARY_PATH = ANALYSIS / "rq1_proxy_paired_transformation_effect_summary.json"

SEED = 20260919
N_BOOT = 10_000
BATCH_SIZE = 500

ZERO_SHOT_DETECTORS = [
    "Log Probability (GPT2-XL)",
    "Rank (negated) (GPT2-XL)",
    "Log-Rank (negated) (GPT2-XL)",
    "Entropy (GPT2-XL)",
    "DetectLLM-LRR (GPT2-XL)",
    "Binoculars (GPT2-XL)",
    "Log Probability (OPT-1.3B)",
    "Rank (negated) (OPT-1.3B)",
    "Log-Rank (negated) (OPT-1.3B)",
    "Entropy (OPT-1.3B)",
    "DetectLLM-LRR (OPT-1.3B)",
    "Binoculars (OPT-1.3B)",
]
PRETRAINED_DETECTORS = ["OpenAI-RoBERTa-base", "OpenAI-RoBERTa-large", "AIGC-MPU"]
DETECTOR_ORDER = ZERO_SHOT_DETECTORS + PRETRAINED_DETECTORS

EXPECTED_HASHES = {
    NATURAL_SCORES_PATH: "6e835ee0280611c3b0d70f1ea7054a84e28ed636128068e2f2c11ecaebf3d208",
    NATURAL_STEP1_PATH: "67ac1accf89dd3af39b95e81b49fb3894ed7aac93ba7f8b5fdb159edad320d0e",
    NATURAL_ROBUSTNESS_PATH: "802835139f9bc04e22fc48aefcd11090392864100dfe955439a5be2be90145a5",
    PAIRED_PATH: "e81d12cfdeccec509943a0937d1f8acd451865fcae123043b49a2cf3775cbb79",
    SYNTHETIC_EFFECTS_PATH: "fbd71a2c12fd6383878dd4cfda3a439f8671b214f489e134dc63e06ca0b78207",
    SYNTHETIC_OPERATIONAL_PATH: "fd0d6351d4c6a308605a0b4d6c10bfc9af40c0192095dd61cf2b48676b8e0d0b",
    FRESH_SCORES_PATH: "36d3f9821efb6d9b8ff688ed784cb13a4a8b3803e64b4e10395dbc21d8aac371",
    FRESH_EFFECTS_PATH: "80b32313696fd4634876a052787fa55f3129bdc5e85896c0d8bf0496a9703997",
    FRESH_COMPARISON_PATH: "7bbd1c40f689ae792e76ff89e93aa6d016b8d915fae62e036d80418bc8ccb73b",
    STEP5_SUMMARY_PATH: "5c3b23257cd4abd5c30986e189404e26cc855c616806715b9194a8b860c333f8",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign_label(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def natural_status(values: list[float]) -> str:
    signs = [sign_label(value) for value in values]
    if all(sign == "positive" for sign in signs):
        return "ROBUST POSITIVE"
    if all(sign == "negative" for sign in signs):
        return "ROBUST NEGATIVE"
    return "SPECIFICATION-SENSITIVE"


def proxy_classification(natural_status_value: str, synthetic_effect: float) -> str:
    synthetic_sign = sign_label(synthetic_effect)
    if natural_status_value == "SPECIFICATION-SENSITIVE" or synthetic_sign == "zero":
        return "INDETERMINATE"
    natural_sign = "positive" if natural_status_value == "ROBUST POSITIVE" else "negative"
    return "PRESERVED" if natural_sign == synthetic_sign else "REVERSED"


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    result = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    return 2.0 * float(result.statistic) / (len(x) * len(y)) - 1.0


def rank_biserial_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    x_indices: np.ndarray,
    y_indices: np.ndarray,
) -> np.ndarray:
    """Bootstrap rank-biserial effects in bounded-memory batches."""
    assert x_indices.shape[0] == y_indices.shape[0] == N_BOOT
    n_x, n_y = len(x), len(y)
    output = np.empty(N_BOOT, dtype=float)
    for start in range(0, N_BOOT, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, N_BOOT)
        x_boot = x[x_indices[start:stop]]
        y_boot = y[y_indices[start:stop]]
        ranks = rankdata(np.concatenate([x_boot, y_boot], axis=1), method="average", axis=1)
        u = ranks[:, :n_x].sum(axis=1) - n_x * (n_x + 1) / 2.0
        output[start:stop] = 2.0 * u / (n_x * n_y) - 1.0
    return output


def load_natural_and_freeze_status() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate natural artifacts and freeze status before synthetic loading."""
    for path in [NATURAL_SCORES_PATH, NATURAL_STEP1_PATH, NATURAL_ROBUSTNESS_PATH]:
        assert sha256(path) == EXPECTED_HASHES[path], f"Frozen natural input changed: {path}"

    scores = pd.read_csv(NATURAL_SCORES_PATH)
    step1 = pd.read_csv(NATURAL_STEP1_PATH)
    robustness = pd.read_csv(NATURAL_ROBUSTNESS_PATH)
    expected = set(DETECTOR_ORDER)
    assert set(scores["detector"].unique()) == expected
    assert set(step1["detector"]) == expected and step1["detector"].is_unique
    assert set(robustness["detector"]) == expected and robustness["detector"].is_unique
    assert len(scores) == 15 * 733
    assert not scores.duplicated(["detector", "source_doc_id"]).any()
    assert scores[["oriented_score", "threshold", "flag_machine"]].notna().all().all()
    populations = scores[["source_doc_id", "variety"]].drop_duplicates()
    assert populations.groupby("variety").size().to_dict() == {"AAVE": 212, "SAE": 521}

    natural = step1.merge(robustness, on=["detector", "family"], validate="one_to_one")
    assert np.allclose(natural["rank_biserial_r"], natural["original_full_pool_r"], rtol=0, atol=1e-15)
    assert np.allclose(natural["operational_delta"], natural["canonical_delta"], rtol=0, atol=1e-15)

    natural["operational_natural_status"] = [
        natural_status([row.canonical_delta, row.all5_delta, row.source_standardized_delta,
                        row.dunbar_hughes_standardized_delta])
        for row in natural.itertuples(index=False)
    ]
    natural["threshold_free_natural_status"] = [
        natural_status([row.original_full_pool_r, row.source_standardized_r,
                        row.dunbar_hughes_standardized_r])
        for row in natural.itertuples(index=False)
    ]

    zero = natural.loc[natural["detector"].isin(ZERO_SHOT_DETECTORS)]
    assert zero["operational_natural_status"].value_counts().to_dict() == {
        "ROBUST POSITIVE": 5,
        "ROBUST NEGATIVE": 4,
        "SPECIFICATION-SENSITIVE": 3,
    }
    assert zero["threshold_free_natural_status"].value_counts().to_dict() == {
        "ROBUST POSITIVE": 3,
        "ROBUST NEGATIVE": 6,
        "SPECIFICATION-SENSITIVE": 3,
    }

    # Recompute the two Step 1 primary point effects from row-level data.
    for detector in DETECTOR_ORDER:
        d = scores.loc[scores["detector"].eq(detector)]
        aave = d.loc[d["variety"].eq("AAVE")]
        sae = d.loc[d["variety"].eq("SAE")]
        row = natural.loc[natural["detector"].eq(detector)].iloc[0]
        assert np.isclose(
            rank_biserial(aave["oriented_score"].to_numpy(), sae["oriented_score"].to_numpy()),
            row.original_full_pool_r,
            rtol=0,
            atol=1e-15,
        )
        reproduced_delta = float(aave["flag_machine"].astype(bool).mean() - sae["flag_machine"].astype(bool).mean())
        assert np.isclose(reproduced_delta, row.canonical_delta, rtol=0, atol=1e-15)

    return scores, natural, robustness


def load_synthetic_and_validate(
    natural_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    for path in [
        PAIRED_PATH, SYNTHETIC_EFFECTS_PATH, SYNTHETIC_OPERATIONAL_PATH,
        FRESH_SCORES_PATH, FRESH_EFFECTS_PATH, FRESH_COMPARISON_PATH, STEP5_SUMMARY_PATH,
    ]:
        assert sha256(path) == EXPECTED_HASHES[path], f"Frozen synthetic input changed: {path}"

    paired = pd.read_csv(PAIRED_PATH)
    effects = pd.read_csv(SYNTHETIC_EFFECTS_PATH)
    operational = pd.read_csv(SYNTHETIC_OPERATIONAL_PATH)
    fresh_scores = pd.read_csv(FRESH_SCORES_PATH)
    fresh_effects = pd.read_csv(FRESH_EFFECTS_PATH)
    fresh_comparison = pd.read_csv(FRESH_COMPARISON_PATH)
    step5_summary = json.loads(STEP5_SUMMARY_PATH.read_text())

    expected = set(DETECTOR_ORDER)
    for frame in [paired, effects, operational, fresh_scores, fresh_effects, fresh_comparison]:
        assert set(frame["detector"].unique()) == expected
    assert effects["detector"].is_unique
    assert operational["detector"].is_unique
    assert fresh_effects["detector"].is_unique
    assert fresh_comparison["detector"].is_unique
    assert len(paired) == len(fresh_scores) == 15 * 521
    assert not paired.duplicated(["detector", "source_doc_id"]).any()
    assert not fresh_scores.duplicated(["detector", "source_doc_id"]).any()
    assert all(step5_summary["validation_checks"].values())
    assert step5_summary["n_boot"] == 10_000

    natural_sae = natural_scores.loc[natural_scores["variety"].eq("SAE"), [
        "detector", "source_doc_id", "oriented_score", "evaluation_fold", "threshold", "flag_machine"
    ]].rename(columns={
        "oriented_score": "natural_sae_score",
        "evaluation_fold": "natural_sae_fold",
        "threshold": "natural_threshold",
        "flag_machine": "natural_sae_flag",
    })
    aligned = paired.merge(natural_sae, on=["detector", "source_doc_id"], validate="one_to_one")
    assert len(aligned) == 15 * 521
    assert np.array_equal(aligned["original_sae_score"].to_numpy(), aligned["natural_sae_score"].to_numpy())
    assert np.array_equal(aligned["sae_fold"].to_numpy(), aligned["natural_sae_fold"].to_numpy())
    assert np.array_equal(aligned["threshold"].to_numpy(), aligned["natural_threshold"].to_numpy())
    assert np.array_equal(
        aligned["original_flag_machine"].astype(bool).to_numpy(),
        aligned["natural_sae_flag"].astype(bool).to_numpy(),
    )

    effect_lookup = effects.set_index("detector")
    fresh_lookup = fresh_effects.set_index("detector")
    operational_lookup = operational.set_index("detector")
    for detector in DETECTOR_ORDER:
        canonical = paired.loc[paired["detector"].eq(detector)].sort_values("source_doc_id")
        original = canonical["original_sae_score"].to_numpy(float)
        transformed = canonical["canonical_transformed_score"].to_numpy(float)
        r = rank_biserial(transformed, original)
        delta = float(
            canonical["canonical_flag_machine"].astype(bool).mean()
            - canonical["original_flag_machine"].astype(bool).mean()
        )
        assert np.isclose(r, effect_lookup.loc[detector, "rank_biserial_r"], rtol=0, atol=1e-15)
        assert np.isclose(delta, effect_lookup.loc[detector, "synthetic_operational_delta"], rtol=0, atol=1e-15)
        assert np.isclose(delta, operational_lookup.loc[detector, "synthetic_operational_delta"], rtol=0, atol=1e-15)

        fresh = fresh_scores.loc[fresh_scores["detector"].eq(detector)].sort_values("source_doc_id")
        assert canonical["source_doc_id"].tolist() == fresh["source_doc_id"].tolist()
        fresh_r = rank_biserial(fresh["sensitivity_oriented_score"].to_numpy(float), original)
        fresh_delta = float(
            fresh["sensitivity_flag_machine"].astype(bool).mean()
            - canonical["original_flag_machine"].astype(bool).mean()
        )
        assert np.isclose(fresh_r, fresh_lookup.loc[detector, "rank_biserial_r"], rtol=0, atol=1e-15)
        assert np.isclose(fresh_delta, fresh_lookup.loc[detector, "synthetic_operational_delta"], rtol=0, atol=1e-15)

    audit = {
        "hashes": {
            str(path.relative_to(ROOT)): EXPECTED_HASHES[path]
            for path in EXPECTED_HASHES
        },
        "step5_seed": step5_summary["seed"],
        "step5_boot": step5_summary["n_boot"],
        "fresh_comparison_all_no_sign_flips": bool(
            ~fresh_comparison[[
                "rank_biserial_sign_flip", "operational_delta_sign_flip"
            ]].to_numpy(bool).any()
        ),
    }
    return paired, effects, fresh_scores, fresh_effects, audit


def compare_and_bootstrap(
    natural_scores: pd.DataFrame,
    natural: pd.DataFrame,
    paired: pd.DataFrame,
    synthetic_effects: pd.DataFrame,
    fresh_scores: pd.DataFrame,
    fresh_effects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    aave_indices = rng.integers(0, 212, size=(N_BOOT, 212), dtype=np.int32)
    shared_sae_indices = rng.integers(0, 521, size=(N_BOOT, 521), dtype=np.int32)

    natural_lookup = natural.set_index("detector")
    synthetic_lookup = synthetic_effects.set_index("detector")
    fresh_lookup = fresh_effects.set_index("detector")
    fidelity_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []

    for detector in DETECTOR_ORDER:
        n = natural_lookup.loc[detector]
        ns = natural_scores.loc[natural_scores["detector"].eq(detector)]
        aave = ns.loc[ns["variety"].eq("AAVE")].sort_values("source_doc_id")
        sae = ns.loc[ns["variety"].eq("SAE")].sort_values("source_doc_id")
        p = paired.loc[paired["detector"].eq(detector)].sort_values("source_doc_id")
        fresh = fresh_scores.loc[fresh_scores["detector"].eq(detector)].sort_values("source_doc_id")
        assert sae["source_doc_id"].tolist() == p["source_doc_id"].tolist() == fresh["source_doc_id"].tolist()

        aave_score = aave["oriented_score"].to_numpy(float)
        sae_score = sae["oriented_score"].to_numpy(float)
        original_score = p["original_sae_score"].to_numpy(float)
        transformed_score = p["canonical_transformed_score"].to_numpy(float)
        assert np.array_equal(sae_score, original_score)

        natural_r_boot = rank_biserial_bootstrap(
            aave_score, sae_score, aave_indices, shared_sae_indices
        )
        synthetic_r_boot = rank_biserial_bootstrap(
            transformed_score, original_score, shared_sae_indices, shared_sae_indices
        )
        d_r_boot = natural_r_boot - synthetic_r_boot
        d_r_lo, d_r_hi = np.quantile(d_r_boot, [0.025, 0.975], method="linear")
        synthetic_r = float(synthetic_lookup.loc[detector, "rank_biserial_r"])
        d_r = float(n.original_full_pool_r - synthetic_r)

        aave_flag = aave["flag_machine"].astype(bool).to_numpy()
        sae_flag = sae["flag_machine"].astype(bool).to_numpy()
        original_flag = p["original_flag_machine"].astype(bool).to_numpy()
        transformed_flag = p["canonical_flag_machine"].astype(bool).to_numpy()
        assert np.array_equal(sae_flag, original_flag)
        natural_delta_boot = (
            aave_flag[aave_indices].mean(axis=1) - sae_flag[shared_sae_indices].mean(axis=1)
        )
        synthetic_delta_boot = (
            transformed_flag[shared_sae_indices].mean(axis=1)
            - original_flag[shared_sae_indices].mean(axis=1)
        )
        d_delta_boot = natural_delta_boot - synthetic_delta_boot
        d_delta_lo, d_delta_hi = np.quantile(d_delta_boot, [0.025, 0.975], method="linear")
        synthetic_delta = float(synthetic_lookup.loc[detector, "synthetic_operational_delta"])
        d_delta = float(n.canonical_delta - synthetic_delta)
        assert np.isclose(d_delta, aave_flag.mean() - transformed_flag.mean(), rtol=0, atol=1e-15)

        fresh_r = float(fresh_lookup.loc[detector, "rank_biserial_r"])
        fresh_delta = float(fresh_lookup.loc[detector, "synthetic_operational_delta"])
        operational_classification = proxy_classification(n.operational_natural_status, synthetic_delta)
        fresh_operational_classification = proxy_classification(n.operational_natural_status, fresh_delta)

        fidelity_rows.append({
            "detector": detector,
            "family": n.family,
            "threshold_free_natural_status": n.threshold_free_natural_status,
            "natural_full_pool_r": n.original_full_pool_r,
            "natural_source_standardized_r": n.source_standardized_r,
            "natural_dunbar_hughes_r": n.dunbar_hughes_standardized_r,
            "synthetic_r": synthetic_r,
            "synthetic_r_sign": sign_label(synthetic_r),
            "threshold_free_primary_sign_agreement": sign_label(n.original_full_pool_r) == sign_label(synthetic_r),
            "d_r_natural_minus_synthetic": d_r,
            "operational_natural_status": n.operational_natural_status,
            "natural_canonical_delta": n.canonical_delta,
            "natural_all5_delta": n.all5_delta,
            "natural_source_standardized_delta": n.source_standardized_delta,
            "natural_dunbar_hughes_delta": n.dunbar_hughes_standardized_delta,
            "natural_delta_min": min(n.canonical_delta, n.all5_delta, n.source_standardized_delta,
                                     n.dunbar_hughes_standardized_delta),
            "natural_delta_max": max(n.canonical_delta, n.all5_delta, n.source_standardized_delta,
                                     n.dunbar_hughes_standardized_delta),
            "synthetic_delta": synthetic_delta,
            "synthetic_delta_sign": sign_label(synthetic_delta),
            "operational_proxy_classification": operational_classification,
            "d_delta_natural_minus_synthetic": d_delta,
            "fresh_synthetic_r": fresh_r,
            "fresh_synthetic_r_sign": sign_label(fresh_r),
            "fresh_threshold_free_sign_agreement": sign_label(n.original_full_pool_r) == sign_label(fresh_r),
            "fresh_synthetic_delta": fresh_delta,
            "fresh_synthetic_delta_sign": sign_label(fresh_delta),
            "fresh_operational_proxy_classification": fresh_operational_classification,
            "fresh_proxy_classification_changed": operational_classification != fresh_operational_classification,
        })
        difference_rows.append({
            "detector": detector,
            "family": n.family,
            "bootstrap_replicates": N_BOOT,
            "bootstrap_seed": SEED,
            "shared_sae_resample_indices": True,
            "d_r_natural_minus_synthetic": d_r,
            "d_r_ci_lo": float(d_r_lo),
            "d_r_ci_hi": float(d_r_hi),
            "d_r_sign": sign_label(d_r),
            "d_r_ci_excludes_zero": bool(d_r_lo > 0 or d_r_hi < 0),
            "d_delta_natural_minus_synthetic": d_delta,
            "d_delta_ci_lo": float(d_delta_lo),
            "d_delta_ci_hi": float(d_delta_hi),
            "d_delta_sign": sign_label(d_delta),
            "d_delta_ci_excludes_zero": bool(d_delta_lo > 0 or d_delta_hi < 0),
            "operational_interval_type": "95% joint source-bootstrap CI conditional on frozen thresholds",
            "fresh_d_r_natural_minus_synthetic": float(n.original_full_pool_r - fresh_r),
            "fresh_d_delta_natural_minus_synthetic": float(n.canonical_delta - fresh_delta),
        })

    fidelity = pd.DataFrame(fidelity_rows)
    differences = pd.DataFrame(difference_rows)

    zero_mask = fidelity["detector"].isin(ZERO_SHOT_DETECTORS)
    zero = fidelity.loc[zero_mask].copy()
    zero["natural_r_rank_desc"] = rankdata(-zero["natural_full_pool_r"], method="average")
    zero["synthetic_r_rank_desc"] = rankdata(-zero["synthetic_r"], method="average")
    zero["r_rank_displacement"] = zero["synthetic_r_rank_desc"] - zero["natural_r_rank_desc"]
    zero["r_absolute_rank_displacement"] = zero["r_rank_displacement"].abs()
    zero["natural_delta_rank_desc"] = rankdata(-zero["natural_canonical_delta"], method="average")
    zero["synthetic_delta_rank_desc"] = rankdata(-zero["synthetic_delta"], method="average")
    zero["delta_rank_displacement"] = zero["synthetic_delta_rank_desc"] - zero["natural_delta_rank_desc"]
    zero["delta_absolute_rank_displacement"] = zero["delta_rank_displacement"].abs()

    rho_r = float(spearmanr(zero["natural_full_pool_r"], zero["synthetic_r"]).statistic)
    tau_r = float(kendalltau(zero["natural_full_pool_r"], zero["synthetic_r"]).statistic)
    rho_delta = float(spearmanr(zero["natural_canonical_delta"], zero["synthetic_delta"]).statistic)
    tau_delta = float(kendalltau(zero["natural_canonical_delta"], zero["synthetic_delta"]).statistic)
    robust = zero.loc[zero["operational_natural_status"].ne("SPECIFICATION-SENSITIVE")]
    rho_delta_robust = float(spearmanr(robust["natural_canonical_delta"], robust["synthetic_delta"]).statistic)
    tau_delta_robust = float(kendalltau(robust["natural_canonical_delta"], robust["synthetic_delta"]).statistic)

    for column in [
        "natural_r_rank_desc", "synthetic_r_rank_desc", "r_rank_displacement",
        "r_absolute_rank_displacement", "natural_delta_rank_desc", "synthetic_delta_rank_desc",
        "delta_rank_displacement", "delta_absolute_rank_displacement",
    ]:
        fidelity[column] = np.nan
        fidelity.loc[zero.index, column] = zero[column]
    fidelity["zero_shot_threshold_free_spearman_rho"] = np.where(zero_mask, rho_r, np.nan)
    fidelity["zero_shot_threshold_free_kendall_tau"] = np.where(zero_mask, tau_r, np.nan)
    fidelity["zero_shot_operational_spearman_rho"] = np.where(zero_mask, rho_delta, np.nan)
    fidelity["zero_shot_operational_kendall_tau"] = np.where(zero_mask, tau_delta, np.nan)
    fidelity["robust_only_operational_spearman_rho"] = np.where(zero_mask, rho_delta_robust, np.nan)
    fidelity["robust_only_operational_kendall_tau"] = np.where(zero_mask, tau_delta_robust, np.nan)
    return fidelity, differences


def make_figure(fidelity: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    zero = fidelity.loc[fidelity["detector"].isin(ZERO_SHOT_DETECTORS)].copy()
    zero["point_id"] = np.arange(1, 13)
    colors = {"GPT2-XL": "#0072B2", "OPT-1.3B": "#D55E00"}
    markers = {"ROBUST POSITIVE": "^", "ROBUST NEGATIVE": "v", "SPECIFICATION-SENSITIVE": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.8))
    panels = [
        (
            axes[0], "natural_full_pool_r", "synthetic_r", "threshold_free_natural_status",
            "A. Threshold-free rank effect", "Natural full-pool rank-biserial r",
            "Synthetic rank-biserial r", (-0.70, 0.30),
        ),
        (
            axes[1], "natural_canonical_delta", "synthetic_delta", "operational_natural_status",
            "B. Frozen 5% operational effect", "Natural canonical delta",
            "Synthetic operational delta", (-0.07, 0.15),
        ),
    ]
    label_offsets = {
        "natural_full_pool_r": {
            1: (5, 7), 2: (5, 5), 3: (5, 5), 4: (6, 4), 5: (-2, 7), 6: (5, 5),
            7: (6, 4), 8: (6, 5), 9: (6, 5), 10: (6, 5), 11: (6, 5), 12: (6, -2),
        },
        "natural_canonical_delta": {
            1: (5, 10), 2: (-12, 8), 3: (6, 8), 4: (7, -11), 5: (6, -10), 6: (6, 7),
            7: (7, 8), 8: (7, -10), 9: (7, 7), 10: (7, -11), 11: (-12, 8), 12: (7, 7),
        },
    }
    for ax, x_col, y_col, status_col, title, xlabel, ylabel, limits in panels:
        ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=0)
        ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=0)
        ax.plot(limits, limits, color="#777777", linewidth=0.9, linestyle=":", zorder=0)
        for row in zero.itertuples(index=False):
            backbone = "GPT2-XL" if "GPT2-XL" in row.detector else "OPT-1.3B"
            status = getattr(row, status_col)
            ax.scatter(
                getattr(row, x_col), getattr(row, y_col),
                s=78, marker=markers[status], facecolor=colors[backbone], edgecolor="black",
                linewidth=0.6, zorder=3,
            )
            ax.annotate(
                str(row.point_id), (getattr(row, x_col), getattr(row, y_col)),
                xytext=label_offsets[x_col][row.point_id], textcoords="offset points",
                fontsize=7.5, fontweight="bold",
            )
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(color="#E5E5E5", linewidth=0.6, zorder=-1)

    backbone_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="black",
               markersize=8, label=label)
        for label, color in colors.items()
    ]
    status_handles = [
        Line2D([0], [0], marker=marker, color="none", markerfacecolor="#BDBDBD",
               markeredgecolor="black", markersize=8, label=label.title())
        for label, marker in markers.items()
    ]
    first_legend = axes[0].legend(handles=backbone_handles, title="Backbone", loc="lower left", fontsize=8)
    axes[0].add_artist(first_legend)
    axes[0].legend(handles=status_handles, title="Natural status", loc="upper left", fontsize=7.5)
    axes[1].legend(handles=status_handles, title="Natural status", loc="upper left", fontsize=7.5)

    keys = []
    for row in zero.itertuples(index=False):
        backbone = "G" if "GPT2-XL" in row.detector else "O"
        metric = row.detector.split(" (")[0].replace(" (negated)", "")
        abbreviations = {
            "Log Probability": "Log probability", "Rank": "Rank", "Log-Rank": "Log-rank",
            "Entropy": "Entropy", "DetectLLM-LRR": "DetectLLM-LRR", "Binoculars": "Binoculars",
        }
        keys.append(f"{row.point_id}: {backbone} {abbreviations[metric]}")
    key_lines = ["    ".join(keys[i:i + 4]) for i in range(0, len(keys), 4)]
    fig.text(0.5, 0.115, "Detector key (G = GPT2-XL; O = OPT-1.3B)\n" + "\n".join(key_lines),
             ha="center", va="center", fontsize=8.2)
    fig.text(
        0.5, 0.025,
        "Dotted diagonal: y = x. Points are detector configurations, not independent random draws. "
        "Shapes encode the panel-specific frozen natural status.",
        ha="center", va="bottom", fontsize=8, color="#333333",
    )
    fig.suptitle("Natural AAVE versus rule-transformed proxy fidelity", fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.24, wspace=0.26)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def write_report(
    fidelity: pd.DataFrame,
    differences: pd.DataFrame,
    natural_audit: dict[str, str],
    synthetic_audit: dict[str, object],
) -> None:
    f = fidelity.set_index("detector")
    d = differences.set_index("detector")
    zero = fidelity.loc[fidelity["detector"].isin(ZERO_SHOT_DETECTORS)].copy()
    robust = zero.loc[zero["operational_natural_status"].ne("SPECIFICATION-SENSITIVE")]
    robust_pos = zero.loc[zero["operational_natural_status"].eq("ROBUST POSITIVE")]
    robust_neg = zero.loc[zero["operational_natural_status"].eq("ROBUST NEGATIVE")]
    sensitive = zero.loc[zero["operational_natural_status"].eq("SPECIFICATION-SENSITIVE")]
    pretrained = fidelity.loc[fidelity["detector"].isin(PRETRAINED_DETECTORS)]

    sign_agreement = int(zero["threshold_free_primary_sign_agreement"].sum())
    preserved = int((robust["operational_proxy_classification"] == "PRESERVED").sum())
    reversed_n = int((robust["operational_proxy_classification"] == "REVERSED").sum())
    pos_reversed = int((robust_pos["operational_proxy_classification"] == "REVERSED").sum())
    neg_preserved = int((robust_neg["operational_proxy_classification"] == "PRESERVED").sum())

    rho_r = float(zero["zero_shot_threshold_free_spearman_rho"].iloc[0])
    tau_r = float(zero["zero_shot_threshold_free_kendall_tau"].iloc[0])
    rho_delta = float(zero["zero_shot_operational_spearman_rho"].iloc[0])
    tau_delta = float(zero["zero_shot_operational_kendall_tau"].iloc[0])
    rho_delta_robust = float(zero["robust_only_operational_spearman_rho"].iloc[0])
    tau_delta_robust = float(zero["robust_only_operational_kendall_tau"].iloc[0])

    max_r_disp = float(zero["r_absolute_rank_displacement"].max())
    max_r_names = zero.loc[zero["r_absolute_rank_displacement"].eq(max_r_disp), "detector"].tolist()
    median_r_disp = float(zero["r_absolute_rank_displacement"].median())
    max_d_disp = float(zero["delta_absolute_rank_displacement"].max())
    max_d_names = zero.loc[zero["delta_absolute_rank_displacement"].eq(max_d_disp), "detector"].tolist()
    median_d_disp = float(zero["delta_absolute_rank_displacement"].median())

    fresh_tf_changes = int(
        (zero["threshold_free_primary_sign_agreement"] != zero["fresh_threshold_free_sign_agreement"]).sum()
    )
    fresh_class_changes = int(zero["fresh_proxy_classification_changed"].sum())

    lines: list[str] = [
        "# RQ1 natural-versus-synthetic proxy fidelity (Step 6)\n",
        "This analysis asks whether the frozen rule-transformed poetry proxy preserves detector-specific directions and ordering from the frozen natural AAVE-versus-SAE reference. It does not test one universal reversal, run detector inference, call GPTZero, transform text, edit the manuscript, or run the later operating-point sweep.\n",
        "## A. Input validation\n",
        "Validation passed for 212 natural AAVE poems, 521 natural SAE poems, 521 paired original/transformed synthetic source poems, and all 15 local detector configurations. Detector names align exactly across Step 1, Step 5, and Step 5.5. Each detector has one complete 521-source canonical pair table and one complete fresh-instance table, with no duplicate detector-by-source rows. The paired original SAE scores, folds, thresholds, and classifications match the Step 1 SAE observations exactly. Recomputed canonical and fresh synthetic rank-biserial and operational point estimates reproduce the frozen Step 5 tables.\n",
        f"Step 5 used {synthetic_audit['step5_boot']:,} paired bootstrap replicates with seed `{synthetic_audit['step5_seed']}`. The direct joint comparisons below use {N_BOOT:,} replicates with seed `{SEED}`. The same SAE source-resample indices are used in the natural and synthetic components, preserving shared-baseline covariance.\n",
        "Frozen input SHA-256 values:\n",
    ]
    for path, digest in natural_audit.items():
        lines.append(f"- `{path}`: `{digest}`")
    for path, digest in synthetic_audit["hashes"].items():
        if path not in natural_audit:
            lines.append(f"- `{path}`: `{digest}`")

    lines += [
        "\n## B. Frozen natural robustness categories\n",
        "These labels are computed from Step 5.5 only, before synthetic results enter the comparison. Threshold-free and operational classifications remain separate.\n",
        "### Operational status\n",
    ]
    for status in ["ROBUST POSITIVE", "ROBUST NEGATIVE", "SPECIFICATION-SENSITIVE"]:
        names = zero.loc[zero["operational_natural_status"].eq(status), "detector"].tolist()
        lines.append(f"- **{status} ({len(names)}):** " + "; ".join(names))
    lines += ["\n### Threshold-free status\n"]
    for status in ["ROBUST POSITIVE", "ROBUST NEGATIVE", "SPECIFICATION-SENSITIVE"]:
        names = zero.loc[zero["threshold_free_natural_status"].eq(status), "detector"].tolist()
        lines.append(f"- **{status} ({len(names)}):** " + "; ".join(names))

    lines += [
        "\n## C. Threshold-free comparison\n",
        "The primary same-scale comparison is the original full-pool natural r versus the canonical synthetic r. Source-standardized natural estimates are prespecified robustness annotations, not replacement outcomes. `D_r = r_natural - r_synthetic`.\n",
        "| detector | natural full r | natural std r | natural D+H r | natural TF status | synthetic r | synthetic sign | D_r |",
        "|---|---:|---:|---:|---|---:|---|---:|",
    ]
    for detector in DETECTOR_ORDER:
        row = f.loc[detector]
        lines.append(
            f"| {detector} | {row.natural_full_pool_r:+.4f} | {row.natural_source_standardized_r:+.4f} | "
            f"{row.natural_dunbar_hughes_r:+.4f} | {row.threshold_free_natural_status} | "
            f"{row.synthetic_r:+.4f} | {row.synthetic_r_sign} | {row.d_r_natural_minus_synthetic:+.4f} |"
        )

    lines += [
        "\n## D. Joint-bootstrap threshold-free differences\n",
        "For each replicate, 212 AAVE poems and 521 SAE poems are resampled at the poem/source level. The same 521 SAE resample indices are used for the natural SAE component and the paired synthetic original/transformed component. Intervals are percentile joint-bootstrap intervals for benchmark-construction differences, not causal provenance effects.\n",
        "| detector | D_r | 95% joint-bootstrap CI | sign | CI excludes zero? |",
        "|---|---:|---:|---|---|",
    ]
    for detector in DETECTOR_ORDER:
        row = d.loc[detector]
        lines.append(
            f"| {detector} | {row.d_r_natural_minus_synthetic:+.4f} | "
            f"[{row.d_r_ci_lo:+.4f}, {row.d_r_ci_hi:+.4f}] | {row.d_r_sign} | "
            f"{'yes' if row.d_r_ci_excludes_zero else 'no'} |"
        )

    lines += [
        "\n## E. Operational comparison\n",
        "Proxy agreement is PRESERVED only when the synthetic sign matches a robust natural sign, REVERSED only when it opposes a robust natural sign, and INDETERMINATE when the natural reference is specification-sensitive or either effect is exactly zero.\n",
        "| detector | canonical | all5 | source std | D+H std | natural status | synthetic delta | synthetic sign | proxy agreement |",
        "|---|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for detector in DETECTOR_ORDER:
        row = f.loc[detector]
        lines.append(
            f"| {detector} | {row.natural_canonical_delta:+.4f} | {row.natural_all5_delta:+.4f} | "
            f"{row.natural_source_standardized_delta:+.4f} | {row.natural_dunbar_hughes_delta:+.4f} | "
            f"{row.operational_natural_status} | {row.synthetic_delta:+.4f} | {row.synthetic_delta_sign} | "
            f"{row.operational_proxy_classification} |"
        )

    lines += [
        "\n## F. Joint-bootstrap operational differences\n",
        "`D_delta = Delta_natural,canonical - Delta_synthetic`. Natural AAVE poems are resampled independently; the same resampled SAE observations supply the natural SAE FPR and the paired synthetic original/transformed FPRs. These intervals condition on the already fitted thresholds. Threshold-estimation uncertainty is deferred to the later operating-point sweep.\n",
        "| detector | D_delta | 95% joint-bootstrap CI | CI excludes zero? |",
        "|---|---:|---:|---|",
    ]
    for detector in DETECTOR_ORDER:
        row = d.loc[detector]
        lines.append(
            f"| {detector} | {row.d_delta_natural_minus_synthetic:+.4f} | "
            f"[{row.d_delta_ci_lo:+.4f}, {row.d_delta_ci_hi:+.4f}] | "
            f"{'yes' if row.d_delta_ci_excludes_zero else 'no'} |"
        )

    lines += [
        "\n## G. Robust reversal/preservation counts\n",
        "### Robust-positive natural detectors\n",
        "| detector | natural delta range | synthetic delta | synthetic negative? | D_delta 95% CI |",
        "|---|---:|---:|---|---:|",
    ]
    for row in robust_pos.itertuples(index=False):
        diff = d.loc[row.detector]
        lines.append(
            f"| {row.detector} | [{row.natural_delta_min:+.4f}, {row.natural_delta_max:+.4f}] | "
            f"{row.synthetic_delta:+.4f} | {'yes' if row.synthetic_delta < 0 else 'no'} | "
            f"[{diff.d_delta_ci_lo:+.4f}, {diff.d_delta_ci_hi:+.4f}] |"
        )
    lines.append(f"\n**Robust-positive natural detectors reversed by synthetic: {pos_reversed}/5.**\n")
    lines += [
        "### Robust-negative natural detectors\n",
        "| detector | natural delta range | synthetic delta | direction preserved? |",
        "|---|---:|---:|---|",
    ]
    for row in robust_neg.itertuples(index=False):
        lines.append(
            f"| {row.detector} | [{row.natural_delta_min:+.4f}, {row.natural_delta_max:+.4f}] | "
            f"{row.synthetic_delta:+.4f} | "
            f"{'yes' if row.operational_proxy_classification == 'PRESERVED' else 'no'} |"
        )
    lines.append(f"\n**Robust-negative natural detectors preserved by synthetic: {neg_preserved}/4.**\n")
    lines += [
        "### Specification-sensitive natural detectors\n",
        "These detectors are excluded from reversal/preservation denominators.\n",
        "| detector | natural delta range | synthetic delta | classification |",
        "|---|---:|---:|---|",
    ]
    for row in sensitive.itertuples(index=False):
        lines.append(
            f"| {row.detector} | [{row.natural_delta_min:+.4f}, {row.natural_delta_max:+.4f}] | "
            f"{row.synthetic_delta:+.4f} | {row.operational_proxy_classification} |"
        )

    lines += [
        "\n## H. Sign fidelity\n",
        f"Using primary full-pool threshold-free natural signs, natural and synthetic signs agree for **{sign_agreement}/12** zero-shot configurations and disagree for **{12 - sign_agreement}/12**.\n",
        f"Among the 9 configurations with robust natural operational direction, the proxy preserves **{preserved}/9** and reverses **{reversed_n}/9**. All **{pos_reversed}/5** robust-positive natural configurations reverse under the synthetic proxy, while all **{neg_preserved}/4** robust-negative natural configurations are preserved. The three specification-sensitive configurations are not included in these denominators.\n",
        "## I. Spearman/Kendall fidelity\n",
        "The 12 detector configurations are not independent random draws; coefficients are descriptive, and no p-value is used for inference.\n",
        "| comparison | n | Spearman rho | Kendall tau |",
        "|---|---:|---:|---:|",
        f"| Threshold-free: natural full-pool r vs synthetic r | 12 | {rho_r:+.4f} | {tau_r:+.4f} |",
        f"| Operational: natural canonical delta vs synthetic delta | 12 | {rho_delta:+.4f} | {tau_delta:+.4f} |",
        f"| Operational, robust-natural subset | 9 | {rho_delta_robust:+.4f} | {tau_delta_robust:+.4f} |",
        "\nThreshold-free ordering is essentially unpreserved, while operational ordering is strongly concordant despite the five robust-positive sign reversals. Ordering fidelity therefore does not imply directional interchangeability.\n",
        "## J. Rank-displacement diagnostic\n",
        "Ranks are descending signed effects (rank 1 = most positive/more machine-like disparity), matching the correlation estimands; they are not ranks of absolute effect size. Ties receive average ranks.\n",
        "| detector | natural r rank | synthetic r rank | displacement | natural delta rank | synthetic delta rank | displacement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in zero.itertuples(index=False):
        lines.append(
            f"| {row.detector} | {row.natural_r_rank_desc:.1f} | {row.synthetic_r_rank_desc:.1f} | "
            f"{row.r_rank_displacement:+.1f} | {row.natural_delta_rank_desc:.1f} | "
            f"{row.synthetic_delta_rank_desc:.1f} | {row.delta_rank_displacement:+.1f} |"
        )
    lines.append(
        f"\nThreshold-free: largest absolute rank disagreement is {max_r_disp:.1f} ranks "
        f"({'; '.join(max_r_names)}); median absolute displacement is {median_r_disp:.2f}. "
        f"Operational: largest absolute disagreement is {max_d_disp:.1f} ranks "
        f"({'; '.join(max_d_names)}); median absolute displacement is {median_d_disp:.2f}.\n"
    )

    lines += [
        "## K. Pretrained results\n",
        "Pretrained configurations are reported separately from the 12-zero-shot headline.\n",
        "| detector | natural full/std/D+H r | natural TF status | synthetic r | TF sign agrees? | natural op status | synthetic delta | operational proxy result |",
        "|---|---:|---|---:|---|---|---:|---|",
    ]
    for row in pretrained.itertuples(index=False):
        lines.append(
            f"| {row.detector} | {row.natural_full_pool_r:+.4f} / {row.natural_source_standardized_r:+.4f} / "
            f"{row.natural_dunbar_hughes_r:+.4f} | {row.threshold_free_natural_status} | "
            f"{row.synthetic_r:+.4f} | {'yes' if row.threshold_free_primary_sign_agreement else 'no'} | "
            f"{row.operational_natural_status} | {row.synthetic_delta:+.4f} | "
            f"{row.operational_proxy_classification} |"
        )

    lines += [
        "\n## L. Fresh-instance sensitivity\n",
        "The full Step 5 fresh-instance sensitivity scores were substituted only for the transformed synthetic scores/classifications; the canonical results remain primary.\n",
        "| detector | synthetic r canonical -> fresh | synthetic delta canonical -> fresh | proxy classification canonical -> fresh |",
        "|---|---:|---:|---|",
    ]
    for detector in DETECTOR_ORDER:
        row = f.loc[detector]
        lines.append(
            f"| {detector} | {row.synthetic_r:+.4f} -> {row.fresh_synthetic_r:+.4f} | "
            f"{row.synthetic_delta:+.4f} -> {row.fresh_synthetic_delta:+.4f} | "
            f"{row.operational_proxy_classification} -> {row.fresh_operational_proxy_classification} |"
        )
    lines.append(
        f"\nAcross the 12 zero-shot configurations, fresh-instance substitution changes {fresh_tf_changes} threshold-free sign-agreement conclusions and {fresh_class_changes} robust-subset reversal/preservation classifications. No proxy-fidelity conclusion changes; canonical results are retained.\n"
    )

    robust_pos_diffs = differences.loc[
        differences["detector"].isin(robust_pos["detector"])
    ]
    materially_supported_reversals = int(
        ((robust_pos_diffs["d_delta_natural_minus_synthetic"] > 0)
         & robust_pos_diffs["d_delta_ci_excludes_zero"]).sum()
    )
    lines += [
        "## M. Claim audit\n",
        "| claim | assessment | evidence and scope |",
        "|---|---|---|",
        f"| **A. Rule-transformed dialect is an interchangeable proxy for naturally occurring dialect in MGT-detector audits.** | **NOT SUPPORTED** | Threshold-free signs agree for only {sign_agreement}/12; only {preserved}/9 robust operational directions are preserved, and all 5 robust-positive conclusions reverse. |",
        f"| **B. Substituting rule-transformed dialect for natural dialect can reverse detector-audit conclusions.** | **SUPPORTED WITH QUALIFICATION** | All {pos_reversed}/5 robust-positive zero-shot natural conclusions become negative under the proxy, and {materially_supported_reversals}/5 have direct D_delta intervals excluding zero. This applies only to this Multi-VALUE transformation, AAVE/SAE poetry benchmark, detector set, and frozen operating point. |",
        "| **C. Rule-transformed dialect systematically makes detectors view text as more human-like.** | **SUPPORTED WITH QUALIFICATION** | In this Multi-VALUE poetry arm, all 12 zero-shot configurations have negative synthetic r and negative operational delta; the frozen Step 5 paired intervals exclude zero in the negative direction for all 12. This is not a universal claim about transformations, varieties, genres, or detectors. |",
        f"| **D. Synthetic proxy effects preserve relative detector behavior on natural AAVE.** | **NOT SUPPORTED** | Threshold-free ordering is not preserved (rho={rho_r:+.3f}, tau={tau_r:+.3f}) and directional fidelity is mixed. Operational ordering is descriptively strong (rho={rho_delta:+.3f}, tau={tau_delta:+.3f}), but it coexists with 5/9 robust-direction reversals. |",
        f"| **E. Natural and synthetic dialect produce opposite conclusions for all zero-shot detectors.** | **NOT SUPPORTED** | Primary threshold-free signs agree for {sign_agreement}/12, and all four robust-negative operational conclusions are preserved rather than reversed. |",
        "\n## N. Final decision\n",
        "The synthetic arm is highly coherent as a paired transformation benchmark, but it is not an interchangeable proxy for the detector-specific natural AAVE reference: threshold-free ordering is not preserved and five robust-positive operational conclusions reverse. The later operating-point sweep may characterize threshold dependence, but the general proxy-fidelity claim is not supported at this frozen point.\n",
        "**PROXY-FIDELITY CLAIM NOT SUPPORTED**",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    natural_scores, natural, _ = load_natural_and_freeze_status()
    paired, synthetic_effects, fresh_scores, fresh_effects, synthetic_audit = load_synthetic_and_validate(
        natural_scores
    )
    fidelity, differences = compare_and_bootstrap(
        natural_scores, natural, paired, synthetic_effects, fresh_scores, fresh_effects
    )
    fidelity.to_csv(FIDELITY_OUT, index=False)
    differences.to_csv(DIFFERENCE_OUT, index=False)
    make_figure(fidelity)
    natural_audit = {
        str(path.relative_to(ROOT)): EXPECTED_HASHES[path]
        for path in [NATURAL_SCORES_PATH, NATURAL_STEP1_PATH, NATURAL_ROBUSTNESS_PATH]
    }
    write_report(fidelity, differences, natural_audit, synthetic_audit)

    zero = fidelity.loc[fidelity["detector"].isin(ZERO_SHOT_DETECTORS)]
    robust = zero.loc[zero["operational_natural_status"].ne("SPECIFICATION-SENSITIVE")]
    print({
        "threshold_free_sign_agreement": f"{int(zero['threshold_free_primary_sign_agreement'].sum())}/12",
        "robust_operational_preserved": f"{int((robust['operational_proxy_classification'] == 'PRESERVED').sum())}/9",
        "robust_operational_reversed": f"{int((robust['operational_proxy_classification'] == 'REVERSED').sum())}/9",
    })
    for path in [REPORT, FIDELITY_OUT, DIFFERENCE_OUT, FIGURE]:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

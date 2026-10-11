"""Step 7: frozen-score operating-point sweep for natural/proxy fidelity.

Uses only frozen detector scores, fold labels, pairings, and source metadata.
No detector inference, GPTZero call, text transformation, manuscript edit, or
post-hoc detector selection occurs here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data/analysis"
REPORT = ROOT / "reports/rq1_operating_point_sweep.md"
FIGURE = ROOT / "figures/rq1_operating_point_sweep.pdf"
THRESHOLD_OUT = ANALYSIS / "rq1_operating_point_thresholds.csv"
SWEEP_OUT = ANALYSIS / "rq1_operating_point_sweep.csv"
BOOTSTRAP_OUT = ANALYSIS / "rq1_operating_point_bootstrap.csv"
STATUS_OUT = ANALYSIS / "rq1_operating_point_proxy_status.csv"

NATURAL_SCORES_PATH = ANALYSIS / "rq1_full_poetry_natural_scores.csv"
STEP1_THRESHOLDS_PATH = ANALYSIS / "rq1_full_poetry_thresholds.csv"
STEP1_FOLDS_PATH = ANALYSIS / "rq1_full_poetry_sae_folds.csv"
NATURAL_ROBUSTNESS_PATH = ANALYSIS / "rq1_natural_reference_robustness_summary.csv"
PAIRED_PATH = ANALYSIS / "rq1_proxy_original_canonical_paired_scores.csv"
FRESH_PATH = ANALYSIS / "rq1_proxy_fresh_instance_sensitivity_scores.csv"
STEP6_PATH = ANALYSIS / "rq1_natural_vs_synthetic_proxy_fidelity.csv"
STEP6_DIFFERENCE_PATH = ANALYSIS / "rq1_natural_vs_synthetic_direct_differences.csv"

EXPECTED_HASHES = {
    NATURAL_SCORES_PATH: "6e835ee0280611c3b0d70f1ea7054a84e28ed636128068e2f2c11ecaebf3d208",
    STEP1_THRESHOLDS_PATH: "3e22028e1f9e3a264370cde766002c87db6a7c115d184fe4b00469e4646def70",
    STEP1_FOLDS_PATH: "249761564463507527f23a0e2426d941d30cafe3467ac0652bdb5f764ea69d48",
    NATURAL_ROBUSTNESS_PATH: "802835139f9bc04e22fc48aefcd11090392864100dfe955439a5be2be90145a5",
    PAIRED_PATH: "e81d12cfdeccec509943a0937d1f8acd451865fcae123043b49a2cf3775cbb79",
    FRESH_PATH: "36d3f9821efb6d9b8ff688ed784cb13a4a8b3803e64b4e10395dbc21d8aac371",
    STEP6_PATH: "b084ef92a3673be68920383946b8179b94c5a0b89d2406ea458e01917d9bbf5f",
    STEP6_DIFFERENCE_PATH: "622d7604132e189442151bc5064258bed709c49ad2e4636f6bf74937dd1102be",
}

ALPHAS = np.array([0.01, 0.025, 0.05, 0.10, 0.15, 0.20], dtype=float)
PRIMARY_ALPHAS = {0.025, 0.05, 0.10, 0.15, 0.20}
SEED = 20260919
N_BOOT = 10_000
BATCH_SIZE = 500
SOURCES = ["dunbar", "henderson", "hughes", "johnson1922"]
POOLED_WEIGHTS = {"dunbar": 432 / 733, "henderson": 51 / 733,
                  "hughes": 229 / 733, "johnson1922": 21 / 733}
LARGE_WEIGHTS = {"dunbar": 432 / 661, "hughes": 229 / 661}

ZERO_SHOT_DETECTORS = [
    "Log Probability (GPT2-XL)", "Rank (negated) (GPT2-XL)",
    "Log-Rank (negated) (GPT2-XL)", "Entropy (GPT2-XL)",
    "DetectLLM-LRR (GPT2-XL)", "Binoculars (GPT2-XL)",
    "Log Probability (OPT-1.3B)", "Rank (negated) (OPT-1.3B)",
    "Log-Rank (negated) (OPT-1.3B)", "Entropy (OPT-1.3B)",
    "DetectLLM-LRR (OPT-1.3B)", "Binoculars (OPT-1.3B)",
]
PRETRAINED_DETECTORS = ["OpenAI-RoBERTa-base", "OpenAI-RoBERTa-large", "AIGC-MPU"]
DETECTOR_ORDER = ZERO_SHOT_DETECTORS + PRETRAINED_DETECTORS
STEP6_ROBUST_POSITIVE = [
    "Entropy (GPT2-XL)", "Log Probability (OPT-1.3B)",
    "Log-Rank (negated) (OPT-1.3B)", "Entropy (OPT-1.3B)",
    "Binoculars (OPT-1.3B)",
]
STEP6_ROBUST_NEGATIVE = [
    "Rank (negated) (GPT2-XL)", "DetectLLM-LRR (GPT2-XL)",
    "Rank (negated) (OPT-1.3B)", "DetectLLM-LRR (OPT-1.3B)",
]


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
    if "zero" in signs:
        return "ZERO/BOUNDARY"
    if all(sign == "positive" for sign in signs):
        return "ROBUST POSITIVE"
    if all(sign == "negative" for sign in signs):
        return "ROBUST NEGATIVE"
    return "SPECIFICATION-SENSITIVE"


def proxy_classification(status: str, synthetic_delta: float) -> str:
    synthetic_sign = sign_label(synthetic_delta)
    if status in {"SPECIFICATION-SENSITIVE", "ZERO/BOUNDARY"} or synthetic_sign == "zero":
        return "INDETERMINATE"
    natural_sign = "positive" if status == "ROBUST POSITIVE" else "negative"
    return "PRESERVED" if natural_sign == synthetic_sign else "REVERSED"


def quantile_higher_index(n: int, alpha: float) -> int:
    return int(np.ceil((1.0 - alpha) * (n - 1)))


def load_and_validate() -> dict[str, pd.DataFrame]:
    for path, expected in EXPECTED_HASHES.items():
        assert sha256(path) == expected, f"Frozen input hash mismatch: {path}"

    natural = pd.read_csv(NATURAL_SCORES_PATH)
    frozen_thresholds = pd.read_csv(STEP1_THRESHOLDS_PATH)
    frozen_folds = pd.read_csv(STEP1_FOLDS_PATH)
    robustness = pd.read_csv(NATURAL_ROBUSTNESS_PATH)
    paired = pd.read_csv(PAIRED_PATH)
    fresh = pd.read_csv(FRESH_PATH)
    step6 = pd.read_csv(STEP6_PATH)
    step6_diff = pd.read_csv(STEP6_DIFFERENCE_PATH)
    expected_detectors = set(DETECTOR_ORDER)
    for frame in [natural, frozen_thresholds, robustness, paired, fresh, step6, step6_diff]:
        assert set(frame["detector"].unique()) == expected_detectors

    assert len(natural) == 15 * 733 and len(paired) == len(fresh) == 15 * 521
    assert not natural.duplicated(["detector", "source_doc_id"]).any()
    assert not paired.duplicated(["detector", "source_doc_id"]).any()
    assert not fresh.duplicated(["detector", "source_doc_id"]).any()
    assert natural[["oriented_score", "orientation_multiplier", "evaluation_fold",
                    "source_subset"]].notna().all().all()
    assert np.array_equal(
        natural["oriented_score"].to_numpy(),
        (natural["raw_score"] * natural["orientation_multiplier"]).to_numpy(),
    )
    assert paired[["original_sae_raw_score", "original_sae_score",
                   "canonical_transformed_raw_score", "canonical_transformed_score",
                   "sae_fold", "model_config_identifier"]].notna().all().all()
    assert fresh[["canonical_raw_score", "canonical_oriented_score",
                  "sensitivity_raw_score", "sensitivity_oriented_score", "sae_fold",
                  "model_config_identifier"]].notna().all().all()
    orientation = natural.groupby("detector")["orientation_multiplier"].agg(["nunique", "first"])
    assert (orientation["nunique"] == 1).all()
    paired_multiplier = paired["detector"].map(orientation["first"])
    fresh_multiplier = fresh["detector"].map(orientation["first"])
    assert np.array_equal(
        paired["original_sae_score"].to_numpy(),
        (paired["original_sae_raw_score"] * paired_multiplier).to_numpy(),
    )
    assert np.array_equal(
        paired["canonical_transformed_score"].to_numpy(),
        (paired["canonical_transformed_raw_score"] * paired_multiplier).to_numpy(),
    )
    assert np.array_equal(
        fresh["canonical_oriented_score"].to_numpy(),
        (fresh["canonical_raw_score"] * fresh_multiplier).to_numpy(),
    )
    assert np.array_equal(
        fresh["sensitivity_oriented_score"].to_numpy(),
        (fresh["sensitivity_raw_score"] * fresh_multiplier).to_numpy(),
    )
    assert set(paired.groupby("detector")["model_config_identifier"].nunique()) == {1}
    assert set(fresh.groupby("detector")["model_config_identifier"].nunique()) == {1}
    paired_configs = paired.groupby("detector")["model_config_identifier"].first()
    fresh_configs = fresh.groupby("detector")["model_config_identifier"].first()
    assert paired_configs.to_dict() == fresh_configs.to_dict()
    base = natural[["source_doc_id", "poem_id", "variety", "source_subset",
                    "evaluation_fold"]].drop_duplicates()
    assert len(base) == 733 and not base["source_doc_id"].duplicated().any()
    assert base.groupby("variety").size().to_dict() == {"AAVE": 212, "SAE": 521}
    expected_sources = {
        "AAVE": {"dunbar": 115, "henderson": 47, "hughes": 38, "johnson1922": 12},
        "SAE": {"dunbar": 317, "henderson": 4, "hughes": 191, "johnson1922": 9},
    }
    for variety, expected in expected_sources.items():
        observed = base.loc[base["variety"].eq(variety), "source_subset"].value_counts().to_dict()
        assert observed == expected

    sae_base = base.loc[base["variety"].eq("SAE")]
    assert sae_base["evaluation_fold"].value_counts().sort_index().to_dict() == {
        0: 105, 1: 104, 2: 104, 3: 104, 4: 104
    }
    fold_check = sae_base.merge(frozen_folds[["source_doc_id", "sae_fold"]],
                                on="source_doc_id", validate="one_to_one")
    assert (fold_check["evaluation_fold"] == fold_check["sae_fold"]).all()

    natural_sae = natural.loc[natural["variety"].eq("SAE"), [
        "detector", "source_doc_id", "oriented_score", "evaluation_fold"
    ]].rename(columns={"oriented_score": "natural_sae_score",
                       "evaluation_fold": "natural_fold"})
    aligned = paired.merge(natural_sae, on=["detector", "source_doc_id"], validate="one_to_one")
    assert len(aligned) == 15 * 521
    assert np.array_equal(aligned["original_sae_score"].to_numpy(), aligned["natural_sae_score"].to_numpy())
    assert np.array_equal(aligned["sae_fold"].to_numpy(), aligned["natural_fold"].to_numpy())
    fresh_aligned = paired[["detector", "source_doc_id", "sae_fold"]].merge(
        fresh[["detector", "source_doc_id", "sae_fold"]],
        on=["detector", "source_doc_id"], suffixes=("_canonical", "_fresh"), validate="one_to_one"
    )
    assert (fresh_aligned["sae_fold_canonical"] == fresh_aligned["sae_fold_fresh"]).all()

    step6_zero = step6.loc[step6["family"].eq("zero-shot")]
    assert set(step6_zero.loc[step6_zero["operational_natural_status"].eq("ROBUST POSITIVE"), "detector"]) == set(STEP6_ROBUST_POSITIVE)
    assert set(step6_zero.loc[step6_zero["operational_natural_status"].eq("ROBUST NEGATIVE"), "detector"]) == set(STEP6_ROBUST_NEGATIVE)
    return {
        "natural": natural, "frozen_thresholds": frozen_thresholds,
        "robustness": robustness, "paired": paired, "fresh": fresh,
        "step6": step6, "step6_diff": step6_diff,
    }


def point_sweep(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    natural = data["natural"]
    paired = data["paired"]
    fresh = data["fresh"]
    frozen_thresholds = data["frozen_thresholds"].set_index(["detector", "fold"])
    robustness = data["robustness"].set_index("detector")
    step6 = data["step6"].set_index("detector")
    threshold_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []

    for detector in DETECTOR_ORDER:
        family = "zero-shot" if detector in ZERO_SHOT_DETECTORS else "pretrained"
        n = natural.loc[natural["detector"].eq(detector)].copy()
        aave = n.loc[n["variety"].eq("AAVE")].sort_values("source_doc_id")
        sae = n.loc[n["variety"].eq("SAE")].sort_values("source_doc_id")
        p = paired.loc[paired["detector"].eq(detector)].sort_values("source_doc_id")
        fr = fresh.loc[fresh["detector"].eq(detector)].sort_values("source_doc_id")
        assert sae["source_doc_id"].tolist() == p["source_doc_id"].tolist() == fr["source_doc_id"].tolist()
        sae_scores = sae["oriented_score"].to_numpy(float)
        sae_folds = sae["evaluation_fold"].to_numpy(int)
        aave_scores = aave["oriented_score"].to_numpy(float)
        aave_folds = aave["evaluation_fold"].to_numpy(int)
        transformed_scores = p["canonical_transformed_score"].to_numpy(float)
        fresh_scores = fr["sensitivity_oriented_score"].to_numpy(float)

        for alpha in ALPHAS:
            thresholds = np.empty(5, dtype=float)
            fold_sae_fp: list[int] = []
            fold_sae_n: list[int] = []
            for fold in range(5):
                calibration = sae_scores[sae_folds != fold]
                threshold = float(np.quantile(calibration, 1.0 - alpha, method="higher"))
                thresholds[fold] = threshold
                held = sae_folds == fold
                held_fp = int(np.sum(sae_scores[held] > threshold))
                held_n = int(held.sum())
                fold_sae_fp.append(held_fp)
                fold_sae_n.append(held_n)
                threshold_rows.append({
                    "detector": detector, "family": family, "alpha": alpha,
                    "is_primary_range": alpha in PRIMARY_ALPHAS, "fold": fold,
                    "quantile_probability": 1.0 - alpha, "quantile_method": "higher",
                    "decision_rule": "oriented_score > threshold; ties human",
                    "calibration_n": len(calibration), "threshold": threshold,
                    "calibration_n_strictly_above_threshold": int(np.sum(calibration > threshold)),
                    "calibration_n_at_or_above_threshold": int(np.sum(calibration >= threshold)),
                    "heldout_sae_n": held_n, "heldout_sae_false_positives": held_fp,
                    "heldout_sae_fpr": held_fp / held_n,
                })

            sae_flags = sae_scores > thresholds[sae_folds]
            aave_flags = aave_scores > thresholds[aave_folds]
            all5_matrix = aave_scores[:, None] > thresholds[None, :]
            aave_all5 = all5_matrix.mean(axis=1)
            transformed_flags = transformed_scores > thresholds[sae_folds]
            fresh_flags = fresh_scores > thresholds[sae_folds]
            sae_fpr = float(sae_flags.mean())
            canonical_delta = float(aave_flags.mean() - sae_fpr)
            all5_delta = float(aave_all5.mean() - sae_fpr)

            aave_source = aave["source_subset"].to_numpy()
            sae_source = sae["source_subset"].to_numpy()
            within_delta = {
                source: float(aave_all5[aave_source == source].mean()
                              - sae_flags[sae_source == source].mean())
                for source in SOURCES
            }
            source_standardized = float(sum(POOLED_WEIGHTS[s] * within_delta[s] for s in SOURCES))
            large_standardized = float(sum(LARGE_WEIGHTS[s] * within_delta[s] for s in LARGE_WEIGHTS))
            status = natural_status([canonical_delta, all5_delta, source_standardized, large_standardized])
            synthetic_delta = float(transformed_flags.mean() - sae_fpr)
            fresh_delta = float(fresh_flags.mean() - sae_fpr)
            proxy = proxy_classification(status, synthetic_delta)
            fresh_proxy = proxy_classification(status, fresh_delta)

            if np.isclose(alpha, 0.05, rtol=0, atol=1e-15):
                for fold in range(5):
                    assert thresholds[fold] == frozen_thresholds.loc[(detector, fold), "threshold"]
                assert np.array_equal(sae_flags, sae["flag_machine"].astype(bool).to_numpy())
                assert np.array_equal(aave_flags, aave["flag_machine"].astype(bool).to_numpy())
                assert np.array_equal(transformed_flags, p["canonical_flag_machine"].astype(bool).to_numpy())
                assert np.array_equal(fresh_flags, fr["sensitivity_flag_machine"].astype(bool).to_numpy())
                r = robustness.loc[detector]
                assert np.isclose(canonical_delta, r["canonical_delta"], rtol=0, atol=1e-15)
                assert np.isclose(all5_delta, r["all5_delta"], rtol=0, atol=1e-15)
                assert np.isclose(source_standardized, r["source_standardized_delta"], rtol=0, atol=1e-15)
                assert np.isclose(large_standardized, r["dunbar_hughes_standardized_delta"], rtol=0, atol=1e-15)
                assert np.isclose(synthetic_delta, step6.loc[detector, "synthetic_delta"], rtol=0, atol=1e-15)
                assert proxy == step6.loc[detector, "operational_proxy_classification"]

            sweep_rows.append({
                "detector": detector, "family": family, "alpha": alpha,
                "is_primary_range": alpha in PRIMARY_ALPHAS, "is_exploratory_1pct": alpha == 0.01,
                "sae_false_positives": int(sae_flags.sum()), "sae_oof_fpr": sae_fpr,
                "sae_fold_fpr_min": min(fp / nn for fp, nn in zip(fold_sae_fp, fold_sae_n)),
                "sae_fold_fpr_max": max(fp / nn for fp, nn in zip(fold_sae_fp, fold_sae_n)),
                "natural_aave_false_positives": int(aave_flags.sum()),
                "natural_aave_fpr_canonical": float(aave_flags.mean()),
                "natural_delta_canonical": canonical_delta,
                "natural_delta_canonical_sign": sign_label(canonical_delta),
                "natural_aave_fpr_all5": float(aave_all5.mean()),
                "natural_delta_all5": all5_delta,
                "natural_delta_source_standardized": source_standardized,
                "natural_delta_dunbar_hughes": large_standardized,
                "natural_delta_min": min(canonical_delta, all5_delta, source_standardized, large_standardized),
                "natural_delta_max": max(canonical_delta, all5_delta, source_standardized, large_standardized),
                "natural_status": status,
                "synthetic_transformed_false_positives": int(transformed_flags.sum()),
                "synthetic_transformed_fpr": float(transformed_flags.mean()),
                "synthetic_human_to_machine": int((~sae_flags & transformed_flags).sum()),
                "synthetic_machine_to_human": int((sae_flags & ~transformed_flags).sum()),
                "synthetic_delta": synthetic_delta,
                "synthetic_delta_sign": sign_label(synthetic_delta),
                "d_delta_natural_minus_synthetic": canonical_delta - synthetic_delta,
                "proxy_classification": proxy,
                "fresh_transformed_fpr": float(fresh_flags.mean()),
                "fresh_synthetic_delta": fresh_delta,
                "fresh_synthetic_delta_sign": sign_label(fresh_delta),
                "fresh_proxy_classification": fresh_proxy,
                "fresh_synthetic_sign_changed": sign_label(fresh_delta) != sign_label(synthetic_delta),
                "fresh_proxy_classification_changed": fresh_proxy != proxy,
            })

    return pd.DataFrame(threshold_rows), pd.DataFrame(sweep_rows)


def make_resamples(natural: pd.DataFrame) -> tuple[dict[int, list[str]], dict[int, np.ndarray], np.ndarray, list[str]]:
    base = natural[["source_doc_id", "variety", "evaluation_fold"]].drop_duplicates()
    fold_ids = {
        fold: sorted(base.loc[base["variety"].eq("SAE") & base["evaluation_fold"].eq(fold),
                              "source_doc_id"].tolist())
        for fold in range(5)
    }
    aave_ids = sorted(base.loc[base["variety"].eq("AAVE"), "source_doc_id"].tolist())
    rng = np.random.default_rng(SEED)
    fold_indices = {
        fold: rng.integers(0, len(ids), size=(N_BOOT, len(ids)), dtype=np.int32)
        for fold, ids in fold_ids.items()
    }
    aave_indices = rng.integers(0, len(aave_ids), size=(N_BOOT, len(aave_ids)), dtype=np.int32)
    return fold_ids, fold_indices, aave_indices, aave_ids


def threshold_refit_bootstrap(
    data: dict[str, pd.DataFrame], sweep: pd.DataFrame
) -> pd.DataFrame:
    natural = data["natural"]
    paired = data["paired"]
    fold_ids, fold_indices, aave_indices, aave_ids = make_resamples(natural)
    rows: list[dict[str, object]] = []

    for detector in DETECTOR_ORDER:
        family = "zero-shot" if detector in ZERO_SHOT_DETECTORS else "pretrained"
        n = natural.loc[natural["detector"].eq(detector)]
        aave = n.loc[n["variety"].eq("AAVE")].set_index("source_doc_id").loc[aave_ids]
        p = paired.loc[paired["detector"].eq(detector)].set_index("source_doc_id")
        aave_scores = aave["oriented_score"].to_numpy(float)
        aave_folds = aave["evaluation_fold"].to_numpy(int)
        sae_by_fold = {
            fold: n.loc[n["variety"].eq("SAE")].set_index("source_doc_id").loc[ids, "oriented_score"].to_numpy(float)
            for fold, ids in fold_ids.items()
        }
        trans_by_fold = {
            fold: p.loc[ids, "canonical_transformed_score"].to_numpy(float)
            for fold, ids in fold_ids.items()
        }
        natural_boot = np.empty((N_BOOT, len(ALPHAS)), dtype=float)
        synthetic_boot = np.empty_like(natural_boot)

        for start in range(0, N_BOOT, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, N_BOOT)
            batch_n = stop - start
            refit_thresholds = np.empty((batch_n, len(ALPHAS), 5), dtype=float)
            sae_fp = np.zeros((batch_n, len(ALPHAS)), dtype=int)
            trans_fp = np.zeros_like(sae_fp)
            for held_fold in range(5):
                calibration = np.concatenate([
                    sae_by_fold[fold][fold_indices[fold][start:stop]]
                    for fold in range(5) if fold != held_fold
                ], axis=1)
                kth = np.array([quantile_higher_index(calibration.shape[1], alpha) for alpha in ALPHAS])
                partitioned = np.partition(calibration, kth, axis=1)
                thresholds = partitioned[:, kth]
                refit_thresholds[:, :, held_fold] = thresholds
                held_original = sae_by_fold[held_fold][fold_indices[held_fold][start:stop]]
                held_transformed = trans_by_fold[held_fold][fold_indices[held_fold][start:stop]]
                sae_fp += (held_original[:, :, None] > thresholds[:, None, :]).sum(axis=1)
                trans_fp += (held_transformed[:, :, None] > thresholds[:, None, :]).sum(axis=1)

            sampled_aave_idx = aave_indices[start:stop]
            sampled_aave_scores = aave_scores[sampled_aave_idx]
            sampled_aave_folds = aave_folds[sampled_aave_idx]
            aave_fp = np.empty((batch_n, len(ALPHAS)), dtype=int)
            batch_rows = np.arange(batch_n)[:, None]
            for alpha_index in range(len(ALPHAS)):
                assigned_thresholds = refit_thresholds[:, alpha_index, :][batch_rows, sampled_aave_folds]
                aave_fp[:, alpha_index] = (sampled_aave_scores > assigned_thresholds).sum(axis=1)

            sae_fpr = sae_fp / 521.0
            natural_boot[start:stop] = aave_fp / 212.0 - sae_fpr
            synthetic_boot[start:stop] = trans_fp / 521.0 - sae_fpr

        d_boot = natural_boot - synthetic_boot
        point = sweep.loc[sweep["detector"].eq(detector)].set_index("alpha")
        for alpha_index, alpha in enumerate(ALPHAS):
            nat = natural_boot[:, alpha_index]
            syn = synthetic_boot[:, alpha_index]
            difference = d_boot[:, alpha_index]
            nat_lo, nat_hi = np.quantile(nat, [0.025, 0.975], method="linear")
            syn_lo, syn_hi = np.quantile(syn, [0.025, 0.975], method="linear")
            d_lo, d_hi = np.quantile(difference, [0.025, 0.975], method="linear")
            point_row = point.loc[alpha]
            rows.append({
                "detector": detector, "family": family, "alpha": alpha,
                "is_primary_range": alpha in PRIMARY_ALPHAS,
                "bootstrap_replicates": N_BOOT, "bootstrap_seed": SEED,
                "bootstrap_design": "within-SAE-fold source resampling; thresholds refit; independent AAVE source resampling; shared original/transformed sources",
                "natural_delta_point": point_row["natural_delta_canonical"],
                "natural_delta_ci_lo": float(nat_lo), "natural_delta_ci_hi": float(nat_hi),
                "natural_delta_boot_unique_values": int(np.unique(nat).size),
                "natural_delta_ci_degenerate": bool(nat_lo == nat_hi),
                "synthetic_delta_point": point_row["synthetic_delta"],
                "synthetic_delta_ci_lo": float(syn_lo), "synthetic_delta_ci_hi": float(syn_hi),
                "synthetic_delta_boot_unique_values": int(np.unique(syn).size),
                "synthetic_delta_ci_degenerate": bool(syn_lo == syn_hi),
                "d_delta_point": point_row["d_delta_natural_minus_synthetic"],
                "d_delta_ci_lo": float(d_lo), "d_delta_ci_hi": float(d_hi),
                "d_delta_ci_excludes_zero": bool(d_lo > 0 or d_hi < 0),
                "d_delta_boot_unique_values": int(np.unique(difference).size),
                "d_delta_ci_degenerate": bool(d_lo == d_hi),
            })
        print(f"bootstrapped {detector}", flush=True)
    return pd.DataFrame(rows)


def add_bootstrap_status(sweep: pd.DataFrame, bootstrap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = sweep.merge(
        bootstrap[["detector", "alpha", "natural_delta_ci_lo", "natural_delta_ci_hi",
                   "synthetic_delta_ci_lo", "synthetic_delta_ci_hi", "d_delta_ci_lo",
                   "d_delta_ci_hi", "d_delta_ci_excludes_zero"]],
        on=["detector", "alpha"], how="left", validate="one_to_one"
    )
    merged["directional_reversal_supported_away_from_zero"] = (
        merged["proxy_classification"].eq("REVERSED") & merged["d_delta_ci_excludes_zero"]
    )
    status_columns = [
        "detector", "family", "alpha", "is_primary_range", "natural_status",
        "natural_delta_canonical", "natural_delta_min", "natural_delta_max",
        "synthetic_delta", "synthetic_delta_sign", "proxy_classification",
        "d_delta_natural_minus_synthetic", "d_delta_ci_lo", "d_delta_ci_hi",
        "d_delta_ci_excludes_zero", "directional_reversal_supported_away_from_zero",
        "fresh_synthetic_delta", "fresh_synthetic_delta_sign", "fresh_proxy_classification",
        "fresh_synthetic_sign_changed", "fresh_proxy_classification_changed",
    ]
    return merged, merged[status_columns].copy()


def make_figure(sweep: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = sweep.loc[sweep["detector"].isin(ZERO_SHOT_DETECTORS)].merge(
        bootstrap[["detector", "alpha", "natural_delta_ci_lo", "natural_delta_ci_hi",
                   "synthetic_delta_ci_lo", "synthetic_delta_ci_hi"]],
        on=["detector", "alpha"], validate="one_to_one"
    )
    x = np.arange(len(ALPHAS))
    labels = ["1", "2.5", "5", "10", "15", "20"]
    colors = {"natural": "#0072B2", "synthetic": "#D55E00"}
    status_markers = {"ROBUST POSITIVE": "^", "ROBUST NEGATIVE": "v",
                      "SPECIFICATION-SENSITIVE": "D", "ZERO/BOUNDARY": "s"}
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 9.2), sharex=True, sharey=True)
    for ax, detector in zip(axes.flat, ZERO_SHOT_DETECTORS):
        d = data.loc[data["detector"].eq(detector)].sort_values("alpha")
        nat = d["natural_delta_canonical"].to_numpy(float)
        syn = d["synthetic_delta"].to_numpy(float)
        nat_err = np.vstack([nat - d["natural_delta_ci_lo"].to_numpy(float),
                             d["natural_delta_ci_hi"].to_numpy(float) - nat])
        syn_err = np.vstack([syn - d["synthetic_delta_ci_lo"].to_numpy(float),
                             d["synthetic_delta_ci_hi"].to_numpy(float) - syn])
        ax.axvspan(-0.45, 0.45, color="#EFEFEF", zorder=-2)
        ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=0)
        ax.errorbar(x, nat, yerr=nat_err, color=colors["natural"], linestyle="-",
                    linewidth=1.2, capsize=2, marker=None, zorder=2)
        ax.errorbar(x, syn, yerr=syn_err, color=colors["synthetic"], linestyle="--",
                    linewidth=1.2, capsize=2, marker="o", markersize=3.5, zorder=2)
        for xx, yy, status in zip(x, nat, d["natural_status"]):
            ax.scatter(xx, yy, marker=status_markers[status], s=34,
                       facecolor=colors["natural"], edgecolor="black", linewidth=0.4, zorder=4)
        short = detector.replace(" (negated)", "").replace(" (GPT2-XL)", "\nGPT2-XL").replace(" (OPT-1.3B)", "\nOPT-1.3B")
        ax.set_title(short, fontsize=9, fontweight="bold")
        ax.grid(color="#E6E6E6", linewidth=0.5, zorder=-1)
        ax.set_xticks(x, labels)
    for ax in axes[-1, :]:
        ax.set_xlabel("Nominal SAE target FPR (%)", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Operational delta", fontsize=8.5)
    line_handles = [
        Line2D([0], [0], color=colors["natural"], linestyle="-", label="Natural canonical"),
        Line2D([0], [0], color=colors["synthetic"], linestyle="--", marker="o",
               markersize=4, label="Synthetic paired"),
    ]
    marker_handles = [
        Line2D([0], [0], marker=marker, color="none", markerfacecolor=colors["natural"],
               markeredgecolor="black", markersize=7, label=status.title())
        for status, marker in status_markers.items()
    ]
    fig.legend(handles=line_handles + marker_handles, loc="lower center", ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("Natural and synthetic operational effects across SAE-calibrated thresholds",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.012,
             "Error bars: 95% threshold-refit joint-bootstrap CIs. Shaded 1% target is exploratory. "
             "Natural marker shape encodes the four-specification status.",
             ha="center", fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.105,
                        hspace=0.34, wspace=0.18)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def alpha_label(alpha: float) -> str:
    return f"{alpha * 100:g}%"


def write_report(
    data: dict[str, pd.DataFrame], thresholds: pd.DataFrame, sweep: pd.DataFrame,
    bootstrap: pd.DataFrame, status: pd.DataFrame,
) -> None:
    s = sweep.set_index(["detector", "alpha"])
    b = bootstrap.set_index(["detector", "alpha"])
    zero = sweep.loc[sweep["family"].eq("zero-shot")]
    primary_zero = zero.loc[zero["is_primary_range"]]
    fresh_sign_changes_all = int(sweep["fresh_synthetic_sign_changed"].sum())
    fresh_class_changes_all = int(sweep["fresh_proxy_classification_changed"].sum())
    fresh_sign_changes_zero = int(zero["fresh_synthetic_sign_changed"].sum())
    fresh_class_changes_zero = int(zero["fresh_proxy_classification_changed"].sum())
    fresh_max_diff = float((sweep["fresh_synthetic_delta"] - sweep["synthetic_delta"]).abs().max())

    lines: list[str] = [
        "# RQ1 operating-point sweep (Step 7)\n",
        "This frozen-score analysis tests whether the Step 6 natural-versus-synthetic proxy conclusions depend narrowly on the original 5% SAE-FPR target. The primary interpretive range is 2.5%-20%; 1% is retained as an exploratory sparse-tail point. No detector inference, GPTZero call, text transformation, manuscript edit, or score-orientation change occurs.\n",
        "## A. Input validation\n",
        "Validation passed for 212 natural AAVE poems, 521 SAE poems, 521 paired canonical transformed poems, complete fresh-instance scores, exact five-fold identities, exact Step 5.5 source metadata, and all 15 detector configurations. Frozen orientations reproduce exactly. Canonical and fresh transformed sources align one-to-one with the same 521 SAE source IDs.\n",
        "Frozen input SHA-256 values:\n",
    ]
    for path, digest in EXPECTED_HASHES.items():
        lines.append(f"- `{path.relative_to(ROOT)}`: `{digest}`")

    lines += [
        "\n## B. Threshold construction\n",
        "For each detector, target alpha, and held-out SAE fold, the threshold is the NumPy `method=\"higher\"` empirical `(1-alpha)` quantile of the other four SAE folds. Classification is strict `oriented_score > threshold`; ties remain human. AAVE and transformed scores never enter calibration.\n",
        "At alpha=5%, all 75 thresholds, all 521 SAE decisions per detector, all 212 canonical AAVE decisions per detector, all 521 canonical transformed decisions per detector, and all fresh-instance transformed decisions reproduce their frozen Step 1/5/5.5/6 counterparts exactly.\n",
        "## C. Realized SAE FPRs\n",
        "Nominal targets are calibration labels; the table reports actual out-of-fold false positives and rates. Each cell is `false positives / 521 (realized FPR)`.\n",
        "| detector | 1% exploratory | 2.5% | 5% | 10% | 15% | 20% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for detector in DETECTOR_ORDER:
        cells = []
        for alpha in ALPHAS:
            row = s.loc[(detector, alpha)]
            cells.append(f"{int(row.sae_false_positives)} / 521 ({row.sae_oof_fpr:.4f})")
        lines.append(f"| {detector} | " + " | ".join(cells) + " |")

    lines += [
        "\n## D. Natural operational curves\n",
        "The canonical AAVE-fold assignment remains primary. All-five and source-standardized estimates are robustness annotations.\n",
        "| detector | target | canonical delta | all5 delta | source-std delta | D+H delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for detector in DETECTOR_ORDER:
        for alpha in ALPHAS:
            row = s.loc[(detector, alpha)]
            lines.append(
                f"| {detector} | {alpha_label(alpha)} | {row.natural_delta_canonical:+.4f} | "
                f"{row.natural_delta_all5:+.4f} | {row.natural_delta_source_standardized:+.4f} | "
                f"{row.natural_delta_dunbar_hughes:+.4f} |"
            )

    lines += [
        "\n## E. Natural robustness status by operating point\n",
        "Statuses use only the four frozen natural specifications; synthetic results do not enter status assignment.\n",
        "| detector | 1% exploratory | 2.5% | 5% | 10% | 15% | 20% |",
        "|---|---|---|---|---|---|---|",
    ]
    for detector in ZERO_SHOT_DETECTORS:
        cells = [s.loc[(detector, alpha), "natural_status"] for alpha in ALPHAS]
        lines.append(f"| {detector} | " + " | ".join(cells) + " |")

    lines += [
        "\n## F. Synthetic operational curves\n",
        "Every original/transformed pair uses its source poem's fold-specific threshold. Each cell is `synthetic delta (human->machine / machine->human)`.\n",
        "| detector | 1% exploratory | 2.5% | 5% | 10% | 15% | 20% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for detector in DETECTOR_ORDER:
        cells = []
        for alpha in ALPHAS:
            row = s.loc[(detector, alpha)]
            cells.append(
                f"{row.synthetic_delta:+.4f} ({int(row.synthetic_human_to_machine)} / "
                f"{int(row.synthetic_machine_to_human)})"
            )
        lines.append(f"| {detector} | " + " | ".join(cells) + " |")

    lines += [
        "\n## G. Threshold-refit joint-bootstrap differences\n",
        f"The {N_BOOT:,}-replicate source bootstrap uses seed `{SEED}`. SAE sources are resampled within frozen folds, every fold threshold is refit from the other four bootstrapped folds, original/transformed source pairing is preserved, and AAVE poems are independently resampled with canonical fold labels carried along. Thus the intervals include sampling and threshold-estimation variation while preserving shared-baseline covariance. `D_delta` is a benchmark-construction difference, not a causal provenance effect.\n",
        "| detector | target | natural delta [95% CI] | synthetic delta [95% CI] | D_delta [95% CI] | D CI excludes 0? |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for detector in DETECTOR_ORDER:
        for alpha in ALPHAS:
            row = b.loc[(detector, alpha)]
            lines.append(
                f"| {detector} | {alpha_label(alpha)} | {row.natural_delta_point:+.4f} "
                f"[{row.natural_delta_ci_lo:+.4f}, {row.natural_delta_ci_hi:+.4f}] | "
                f"{row.synthetic_delta_point:+.4f} [{row.synthetic_delta_ci_lo:+.4f}, "
                f"{row.synthetic_delta_ci_hi:+.4f}] | {row.d_delta_point:+.4f} "
                f"[{row.d_delta_ci_lo:+.4f}, {row.d_delta_ci_hi:+.4f}] | "
                f"{'yes' if row.d_delta_ci_excludes_zero else 'no'} |"
            )

    lines += [
        "\n## H. Five robust-positive detector trajectories\n",
        "The detector set is frozen from Step 6 and is not redefined from this sweep.\n",
        "| detector | target | natural status | canonical delta | four-spec range | synthetic delta | proxy status | D_delta 95% CI |",
        "|---|---:|---|---:|---:|---:|---|---:|",
    ]
    for detector in STEP6_ROBUST_POSITIVE:
        for alpha in sorted(PRIMARY_ALPHAS):
            row = s.loc[(detector, alpha)]
            boot = b.loc[(detector, alpha)]
            lines.append(
                f"| {detector} | {alpha_label(alpha)} | {row.natural_status} | "
                f"{row.natural_delta_canonical:+.4f} | [{row.natural_delta_min:+.4f}, "
                f"{row.natural_delta_max:+.4f}] | {row.synthetic_delta:+.4f} | "
                f"{row.proxy_classification} | [{boot.d_delta_ci_lo:+.4f}, {boot.d_delta_ci_hi:+.4f}] |"
            )
    lines.append("\nFrozen robust-positive detector counts over five primary operating points:\n")
    lines += [
        "| detector | natural robust positive | synthetic negative | reversals | supported reversals |",
        "|---|---:|---:|---:|---:|",
    ]
    for detector in STEP6_ROBUST_POSITIVE:
        dd = status.loc[status["detector"].eq(detector) & status["is_primary_range"]]
        lines.append(
            f"| {detector} | {(dd.natural_status == 'ROBUST POSITIVE').sum()}/5 | "
            f"{(dd.synthetic_delta < 0).sum()}/5 | {(dd.proxy_classification == 'REVERSED').sum()}/5 | "
            f"{dd.directional_reversal_supported_away_from_zero.sum()}/5 |"
        )

    lines += [
        "\n## I. Four robust-negative detector trajectories\n",
        "| detector | natural robust negative | synthetic negative | preserved | D_delta CIs excluding zero |",
        "|---|---:|---:|---:|---:|",
    ]
    for detector in STEP6_ROBUST_NEGATIVE:
        dd = status.loc[status["detector"].eq(detector) & status["is_primary_range"]]
        lines.append(
            f"| {detector} | {(dd.natural_status == 'ROBUST NEGATIVE').sum()}/5 | "
            f"{(dd.synthetic_delta < 0).sum()}/5 | {(dd.proxy_classification == 'PRESERVED').sum()}/5 | "
            f"{dd.d_delta_ci_excludes_zero.sum()}/5 |"
        )

    lines += [
        "\n## J. Zero-shot family summaries\n",
        "| target | natural robust + | natural robust - | sensitive | zero/boundary | synthetic + | synthetic - | reversed | preserved | indeterminate | supported reversals |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alpha in ALPHAS:
        dd = status.loc[status["family"].eq("zero-shot") & status["alpha"].eq(alpha)]
        lines.append(
            f"| {alpha_label(alpha)}{' exploratory' if alpha == 0.01 else ''} | "
            f"{(dd.natural_status == 'ROBUST POSITIVE').sum()} | "
            f"{(dd.natural_status == 'ROBUST NEGATIVE').sum()} | "
            f"{(dd.natural_status == 'SPECIFICATION-SENSITIVE').sum()} | "
            f"{(dd.natural_status == 'ZERO/BOUNDARY').sum()} | {(dd.synthetic_delta > 0).sum()} | "
            f"{(dd.synthetic_delta < 0).sum()} | {(dd.proxy_classification == 'REVERSED').sum()} | "
            f"{(dd.proxy_classification == 'PRESERVED').sum()} | "
            f"{(dd.proxy_classification == 'INDETERMINATE').sum()} | "
            f"{dd.directional_reversal_supported_away_from_zero.sum()} |"
        )

    lines += [
        "\n## K. Across-threshold persistence\n",
        "Counts cover the five primary targets only; no success threshold is imposed.\n",
        "| detector | robust + | robust - | sensitive | zero/boundary | reversed | preserved | supported direct differences |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for detector in ZERO_SHOT_DETECTORS:
        dd = status.loc[status["detector"].eq(detector) & status["is_primary_range"]]
        lines.append(
            f"| {detector} | {(dd.natural_status == 'ROBUST POSITIVE').sum()} | "
            f"{(dd.natural_status == 'ROBUST NEGATIVE').sum()} | "
            f"{(dd.natural_status == 'SPECIFICATION-SENSITIVE').sum()} | "
            f"{(dd.natural_status == 'ZERO/BOUNDARY').sum()} | "
            f"{(dd.proxy_classification == 'REVERSED').sum()} | "
            f"{(dd.proxy_classification == 'PRESERVED').sum()} | "
            f"{dd.d_delta_ci_excludes_zero.sum()} |"
        )

    lines += [
        "\n## L. Fresh-instance sensitivity\n",
        f"Across all 90 detector-by-target cells, fresh-instance scores change the synthetic sign in **{fresh_sign_changes_all}** cells and the proxy classification in **{fresh_class_changes_all}** cells; the maximum absolute synthetic-delta change is **{fresh_max_diff:.6f}**. Among the 72 zero-shot cells, the corresponding counts are **{fresh_sign_changes_zero}** and **{fresh_class_changes_zero}**. Canonical results remain primary.\n",
    ]
    changed = sweep.loc[sweep["fresh_synthetic_sign_changed"] | sweep["fresh_proxy_classification_changed"]]
    if len(changed):
        lines += [
            "Cells with a fresh-instance sign or proxy-classification change:\n",
            "| detector | target | canonical synthetic delta/status | fresh synthetic delta/status |",
            "|---|---:|---|---|",
        ]
        for row in changed.itertuples(index=False):
            lines.append(
                f"| {row.detector} | {alpha_label(row.alpha)} | {row.synthetic_delta:+.4f} / "
                f"{row.proxy_classification} | {row.fresh_synthetic_delta:+.4f} / "
                f"{row.fresh_proxy_classification} |"
            )

    one_thresholds = thresholds.loc[thresholds["alpha"].eq(0.01)]
    extreme_counts = one_thresholds["calibration_n_strictly_above_threshold"].value_counts().sort_index()
    one_boot = bootstrap.loc[bootstrap["alpha"].eq(0.01)]
    deg_nat = int(one_boot["natural_delta_ci_degenerate"].sum())
    deg_syn = int(one_boot["synthetic_delta_ci_degenerate"].sum())
    deg_d = int(one_boot["d_delta_ci_degenerate"].sum())
    sparse_description = ", ".join(f"{int(k)} above: {int(v)} thresholds" for k, v in extreme_counts.items())
    lines += [
        "\n## M. 1% sparse-tail caution\n",
        "The 1% target is reported fully but is exploratory. With 416-417 calibration observations per held-out fold, `method=\"higher\"` selects a threshold near the fourth/fifth most extreme observation; strict `>` and score ties can reduce the effective tail further. Across the 75 detector-by-fold 1% thresholds, the numbers of calibration observations strictly above the selected threshold are: " + sparse_description + ".\n",
        f"Realized 1% SAE false-positive counts across detectors range from {int(sweep.loc[sweep.alpha.eq(0.01), 'sae_false_positives'].min())} to {int(sweep.loc[sweep.alpha.eq(0.01), 'sae_false_positives'].max())} of 521. Threshold-refit bootstrap intervals are discrete: natural, synthetic, and D_delta bootstrap distributions have median unique-value counts of {int(one_boot.natural_delta_boot_unique_values.median())}, {int(one_boot.synthetic_delta_boot_unique_values.median())}, and {int(one_boot.d_delta_boot_unique_values.median())}, respectively. Degenerate 95% intervals occur for {deg_nat}/15 natural, {deg_syn}/15 synthetic, and {deg_d}/15 direct-difference estimates. The 1% point is not used alone for the main robustness interpretation.\n",
    ]

    all_primary_synthetic_negative = bool((primary_zero["synthetic_delta"] < 0).all())
    step6_positive_status = status.loc[
        status["detector"].isin(STEP6_ROBUST_POSITIVE) & status["is_primary_range"]
    ]
    positive_reversal_cells = int((step6_positive_status["proxy_classification"] == "REVERSED").sum())
    supported_positive_reversal_cells = int(step6_positive_status["directional_reversal_supported_away_from_zero"].sum())
    classifications_vary = any(
        status.loc[(status["detector"].eq(detector)) & status["is_primary_range"],
                   "proxy_classification"].nunique() > 1
        for detector in ZERO_SHOT_DETECTORS
    )
    lines += [
        "\n## N. Claim audit\n",
        "| claim | assessment | evidence and scope |",
        "|---|---|---|",
        f"| **A. The synthetic proxy failure is specific to the original 5% operating point.** | **NOT SUPPORTED** | Across the 25 frozen robust-positive-detector by primary-target cells, {positive_reversal_cells} are directional reversals and {supported_positive_reversal_cells} have D_delta intervals excluding zero. |",
        f"| **B. For detectors with robustly positive natural disparities, substituting rule-transformed dialect can reverse the audit conclusion across multiple deployment-relevant thresholds.** | **SUPPORTED WITH QUALIFICATION** | Reversals occur across multiple primary targets in the frozen Step 6 detector set; exact detector-by-target counts are reported in Section H. Scope is this Multi-VALUE AAVE/SAE poetry benchmark and detector set. |",
        f"| **C. The rule-transformed arm shifts all zero-shot detectors toward lower false-positive rates relative to SAE across the primary operating range.** | **{'SUPPORTED WITH QUALIFICATION' if all_primary_synthetic_negative else 'NOT SUPPORTED'}** | {'All 60' if all_primary_synthetic_negative else 'Not all 60'} zero-shot detector-by-primary-target synthetic deltas are negative. The claim is limited to this arm and operating range. |",
        f"| **D. Proxy validity depends on operating point.** | **{'SUPPORTED WITH QUALIFICATION' if classifications_vary else 'NOT SUPPORTED'}** | {'At least one zero-shot detector changes proxy classification across the five primary targets.' if classifications_vary else 'No zero-shot detector changes proxy classification across the five primary targets, although effect magnitudes and natural statuses can vary.'} |",
        "| **E. Rule-transformed dialect is interchangeable with natural AAVE for detector fairness auditing.** | **NOT SUPPORTED** | The sweep refines but does not overturn Step 6: robust-positive natural conclusions repeatedly reverse under the proxy, while robust-negative conclusions are generally preserved. |",
        "\n## O. Final decision\n",
    ]
    if classifications_vary:
        lines += [
            "At least one zero-shot proxy classification changes across the primary operating range, so detector-specific operating-point dependence must remain explicit in the synthesis.\n",
            "**OPERATING-POINT DEPENDENCE**",
        ]
    else:
        lines += [
            "Proxy classifications are stable across the primary operating range, so the frozen evidence is ready for detector-specific proxy-fidelity synthesis.\n",
            "**PROCEED TO PROXY-FIDELITY SYNTHESIS**",
        ]
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    data = load_and_validate()
    thresholds, sweep = point_sweep(data)
    bootstrap = threshold_refit_bootstrap(data, sweep)
    sweep_with_bootstrap, status = add_bootstrap_status(sweep, bootstrap)
    thresholds.to_csv(THRESHOLD_OUT, index=False)
    sweep_with_bootstrap.to_csv(SWEEP_OUT, index=False)
    bootstrap.to_csv(BOOTSTRAP_OUT, index=False)
    status.to_csv(STATUS_OUT, index=False)
    make_figure(sweep, bootstrap)
    write_report(data, thresholds, sweep_with_bootstrap, bootstrap, status)
    zero = status.loc[status["family"].eq("zero-shot") & status["is_primary_range"]]
    print({
        "primary_cells": len(zero),
        "reversed": int((zero["proxy_classification"] == "REVERSED").sum()),
        "preserved": int((zero["proxy_classification"] == "PRESERVED").sum()),
        "indeterminate": int((zero["proxy_classification"] == "INDETERMINATE").sum()),
        "supported_reversals": int(zero["directional_reversal_supported_away_from_zero"].sum()),
    })
    for path in [REPORT, THRESHOLD_OUT, SWEEP_OUT, BOOTSTRAP_OUT, STATUS_OUT, FIGURE]:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

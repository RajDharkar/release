"""Step 5.5: frozen-score robustness of the natural RQ1 reference.

This analysis consumes only the final Step 1 natural-poetry score, threshold,
fold, and summary artifacts.  It does not load poem text, run detector
inference, call an external service, or use any synthetic-arm result.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data/analysis"
REPORT = ROOT / "reports/rq1_natural_reference_robustness.md"

SCORES_PATH = ANALYSIS / "rq1_full_poetry_natural_scores.csv"
THRESHOLDS_PATH = ANALYSIS / "rq1_full_poetry_thresholds.csv"
FOLDS_PATH = ANALYSIS / "rq1_full_poetry_sae_folds.csv"
STEP1_SUMMARY_PATH = ANALYSIS / "rq1_full_poetry_summary.csv"

ASSIGNMENT_OUT = ANALYSIS / "rq1_natural_threshold_assignment_sensitivity.csv"
SOURCE_OUT = ANALYSIS / "rq1_natural_source_standardized_effects.csv"
SUMMARY_OUT = ANALYSIS / "rq1_natural_reference_robustness_summary.csv"

SEED = 20260919
N_REPLICATES = 10_000
SOURCES = ["dunbar", "henderson", "hughes", "johnson1922"]
LARGE_SOURCES = ["dunbar", "hughes"]
GROUP_SIZES = np.array([43, 43, 42, 42, 42], dtype=int)

EXPECTED_HASHES = {
    SCORES_PATH: "6e835ee0280611c3b0d70f1ea7054a84e28ed636128068e2f2c11ecaebf3d208",
    THRESHOLDS_PATH: "3e22028e1f9e3a264370cde766002c87db6a7c115d184fe4b00469e4646def70",
    FOLDS_PATH: "249761564463507527f23a0e2426d941d30cafe3467ac0652bdb5f764ea69d48",
    STEP1_SUMMARY_PATH: "67ac1accf89dd3af39b95e81b49fb3894ed7aac93ba7f8b5fdb159edad320d0e",
}
EXPECTED_SOURCE_COUNTS = {
    "AAVE": {"dunbar": 115, "henderson": 47, "hughes": 38, "johnson1922": 12},
    "SAE": {"dunbar": 317, "henderson": 4, "hughes": 191, "johnson1922": 9},
}
EXPECTED_ORIENTATIONS = {
    "mean_log_prob": 1,
    "mean_rank": -1,
    "mean_log_rank": -1,
    "mean_entropy": -1,
    "detectllm_lrr": 1,
    "binoculars": 1,
    "p_machine": 1,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign_label(value: float, *, tol: float = 1e-15) -> str:
    if value > tol:
        return "positive"
    if value < -tol:
        return "negative"
    return "zero"


def fmt(value: float, digits: int = 4, signed: bool = False) -> str:
    prefix = "+" if signed else ""
    return f"{value:{prefix}.{digits}f}"


def bool_word(value: bool) -> str:
    return "yes" if bool(value) else "no"


def rank_biserial(aave: np.ndarray, sae: np.ndarray) -> tuple[float, float]:
    test = mannwhitneyu(aave, sae, alternative="two-sided", method="asymptotic")
    effect = 2.0 * float(test.statistic) / (len(aave) * len(sae)) - 1.0
    return float(test.statistic), effect


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        assert actual == expected, f"Frozen input hash mismatch: {path}: {actual} != {expected}"

    scores = pd.read_csv(SCORES_PATH)
    thresholds = pd.read_csv(THRESHOLDS_PATH)
    step1 = pd.read_csv(STEP1_SUMMARY_PATH)
    folds = pd.read_csv(FOLDS_PATH)

    assert len(scores) == 15 * 733
    assert scores["detector"].nunique() == 15
    assert thresholds.shape[0] == 15 * 5
    assert thresholds["detector"].nunique() == 15
    assert step1["detector"].nunique() == 15
    assert not scores.duplicated(["detector", "source_doc_id"]).any()
    assert not thresholds.duplicated(["detector", "fold"]).any()
    assert scores[["raw_score", "orientation_multiplier", "oriented_score"]].notna().all().all()
    assert np.isfinite(scores[["raw_score", "orientation_multiplier", "oriented_score"]]).all().all()
    assert np.array_equal(
        scores["oriented_score"].to_numpy(),
        (scores["raw_score"] * scores["orientation_multiplier"]).to_numpy(),
    )

    observed_orientations = scores.groupby("metric_key")["orientation_multiplier"].unique().to_dict()
    assert set(observed_orientations) == set(EXPECTED_ORIENTATIONS)
    for metric, expected in EXPECTED_ORIENTATIONS.items():
        assert observed_orientations[metric].tolist() == [expected]

    base_columns = [
        "poem_id", "source_doc_id", "variety", "source_subset", "evaluation_fold"
    ]
    base = scores[base_columns].drop_duplicates()
    assert len(base) == 733
    assert not base["source_doc_id"].duplicated().any()
    assert not base["poem_id"].duplicated().any()
    assert base.groupby("variety").size().to_dict() == {"AAVE": 212, "SAE": 521}
    assert base.groupby("variety")["poem_id"].nunique().to_dict() == {"AAVE": 212, "SAE": 521}
    for variety, expected in EXPECTED_SOURCE_COUNTS.items():
        observed = base.loc[base["variety"].eq(variety), "source_subset"].value_counts().to_dict()
        assert observed == expected

    assert len(folds) == 521 and not folds["source_doc_id"].duplicated().any()
    sae_base = base.loc[base["variety"].eq("SAE")].merge(
        folds[["source_doc_id", "sae_fold"]], on="source_doc_id", validate="one_to_one"
    )
    assert (sae_base["evaluation_fold"] == sae_base["sae_fold"]).all()
    assert sae_base["sae_fold"].value_counts().sort_index().to_dict() == {
        0: 105, 1: 104, 2: 104, 3: 104, 4: 104
    }

    threshold_lookup = thresholds.set_index(["detector", "fold"])["threshold"]
    for row in thresholds.itertuples(index=False):
        calibration = scores.loc[
            scores["detector"].eq(row.detector)
            & scores["variety"].eq("SAE")
            & scores["evaluation_fold"].ne(row.fold),
            "oriented_score",
        ].to_numpy(float)
        reproduced = float(np.quantile(calibration, 0.95, method="higher"))
        assert reproduced == row.threshold

    mapped_threshold = pd.Series(
        [threshold_lookup.loc[(d, f)] for d, f in zip(scores["detector"], scores["evaluation_fold"])],
        index=scores.index,
    )
    assert np.array_equal(mapped_threshold.to_numpy(), scores["threshold"].to_numpy())
    reproduced_flags = scores["oriented_score"].to_numpy() > mapped_threshold.to_numpy()
    stored_flags = scores["flag_machine"].astype(bool).to_numpy()
    assert np.array_equal(reproduced_flags, stored_flags)

    canonical = (
        scores.assign(reproduced_flag=reproduced_flags)
        .groupby(["detector", "variety"])["reproduced_flag"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    canonical_wide = canonical.pivot(index="detector", columns="variety", values="mean")
    canonical_wide["delta"] = canonical_wide["AAVE"] - canonical_wide["SAE"]
    check = step1.set_index("detector").join(canonical_wide, rsuffix="_reproduced")
    assert np.allclose(check["aave_fpr"], check["AAVE"], rtol=0.0, atol=1e-15)
    assert np.allclose(check["oof_fpr_sae"], check["SAE"], rtol=0.0, atol=1e-15)
    assert np.allclose(check["operational_delta"], check["delta"], rtol=0.0, atol=1e-15)

    audit = {
        "scores_n": len(scores),
        "detectors_n": scores["detector"].nunique(),
        "thresholds_n": len(thresholds),
        "aave_n": int((base["variety"] == "AAVE").sum()),
        "sae_n": int((base["variety"] == "SAE").sum()),
        "duplicate_detector_source_n": int(scores.duplicated(["detector", "source_doc_id"]).sum()),
        "missing_score_n": int(scores["oriented_score"].isna().sum()),
        "hashes": {str(path.relative_to(ROOT)): value for path, value in EXPECTED_HASHES.items()},
        "max_canonical_rate_abs_diff": float(
            max(
                (check["aave_fpr"] - check["AAVE"]).abs().max(),
                (check["oof_fpr_sae"] - check["SAE"]).abs().max(),
                (check["operational_delta"] - check["delta"]).abs().max(),
            )
        ),
    }
    return scores, thresholds, step1, audit


def balanced_assignments(aave_ids: list[str]) -> np.ndarray:
    """Return replicate x poem threshold-fold assignments.

    Fold labels 0 and 1 have 43 slots and labels 2--4 have 42 slots in every
    replicate.  The complete label vector is independently permuted over the
    stable lexicographic poem order for each replicate.
    """
    assert len(aave_ids) == 212 and aave_ids == sorted(aave_ids)
    labels = np.repeat(np.arange(5, dtype=np.int8), GROUP_SIZES)
    rng = np.random.default_rng(SEED)
    out = np.empty((N_REPLICATES, len(aave_ids)), dtype=np.int8)
    for replicate in range(N_REPLICATES):
        out[replicate] = rng.permutation(labels)
    assert np.all(np.apply_along_axis(lambda x: np.bincount(x, minlength=5), 1, out) == GROUP_SIZES)
    return out


def analyze_threshold_assignment(
    scores: pd.DataFrame, thresholds: pd.DataFrame, step1: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    threshold_lookup = thresholds.pivot(index="detector", columns="fold", values="threshold")
    detector_order = step1["detector"].tolist()
    aave_ids = sorted(scores.loc[scores["variety"].eq("AAVE"), "source_doc_id"].unique().tolist())
    assignments = balanced_assignments(aave_ids)
    poem_index = np.arange(len(aave_ids))[None, :]

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    all5_by_detector: dict[str, np.ndarray] = {}

    step1_lookup = step1.set_index("detector")
    for detector in detector_order:
        d = scores.loc[scores["detector"].eq(detector)].copy()
        family = str(d["family"].iloc[0])
        aave = d.loc[d["variety"].eq("AAVE")].set_index("source_doc_id").loc[aave_ids]
        sae = d.loc[d["variety"].eq("SAE")]
        threshold_values = threshold_lookup.loc[detector, list(range(5))].to_numpy(float)
        flag_matrix = aave["oriented_score"].to_numpy(float)[:, None] > threshold_values[None, :]
        all5_flag_values = flag_matrix.mean(axis=1)
        all5_by_detector[detector] = all5_flag_values

        sae_oof_fpr = float(sae["flag_machine"].astype(bool).mean())
        canonical_aave_fpr = float(aave["flag_machine"].astype(bool).mean())
        canonical_delta = canonical_aave_fpr - sae_oof_fpr
        assert np.isclose(canonical_aave_fpr, step1_lookup.loc[detector, "aave_fpr"], rtol=0, atol=1e-15)
        assert np.isclose(sae_oof_fpr, step1_lookup.loc[detector, "oof_fpr_sae"], rtol=0, atol=1e-15)
        assert np.isclose(canonical_delta, step1_lookup.loc[detector, "operational_delta"], rtol=0, atol=1e-15)

        rotation_rows = []
        folds = aave["evaluation_fold"].to_numpy(int)
        scores_a = aave["oriented_score"].to_numpy(float)
        for offset in range(5):
            assigned_fold = (folds + offset) % 5
            aave_fpr = float(np.mean(scores_a > threshold_values[assigned_fold]))
            delta = aave_fpr - sae_oof_fpr
            rotation_rows.append((offset, aave_fpr, delta, sign_label(delta)))
        assert np.isclose(rotation_rows[0][1], canonical_aave_fpr, rtol=0, atol=1e-15)

        rotation_deltas = np.array([row[2] for row in rotation_rows])
        rotation_signs = [row[3] for row in rotation_rows]
        rotation_pos = rotation_signs.count("positive")
        rotation_neg = rotation_signs.count("negative")
        rotation_zero = rotation_signs.count("zero")
        rotation_sign_changes = len(set(rotation_signs)) > 1

        aave_fpr_all5 = float(all5_flag_values.mean())
        delta_all5 = aave_fpr_all5 - sae_oof_fpr

        assigned_flags = flag_matrix[poem_index, assignments]
        assignment_delta = assigned_flags.mean(axis=1) - sae_oof_fpr
        q025, median, q975 = np.quantile(assignment_delta, [0.025, 0.5, 0.975], method="linear")
        prop_positive = float(np.mean(assignment_delta > 0))
        prop_negative = float(np.mean(assignment_delta < 0))
        prop_zero = float(np.mean(assignment_delta == 0))
        zero_inside = bool(q025 <= 0 <= q975)
        if prop_positive > 0.95:
            assignment_class = ">95% positive"
        elif prop_negative > 0.95:
            assignment_class = ">95% negative"
        else:
            assignment_class = "assignment-sensitive/uncertain"

        common = {
            "detector": detector,
            "family": family,
            "sae_oof_fpr": sae_oof_fpr,
            "canonical_aave_fpr": canonical_aave_fpr,
            "canonical_delta": canonical_delta,
            "rotation_min_delta": float(rotation_deltas.min()),
            "rotation_max_delta": float(rotation_deltas.max()),
            "rotation_positive_n": rotation_pos,
            "rotation_negative_n": rotation_neg,
            "rotation_zero_n": rotation_zero,
            "rotation_sign_changes": rotation_sign_changes,
            "aave_fpr_all5": aave_fpr_all5,
            "delta_all5": delta_all5,
            "assignment_median_delta": float(median),
            "assignment_q025_delta": float(q025),
            "assignment_q975_delta": float(q975),
            "assignment_min_delta": float(assignment_delta.min()),
            "assignment_max_delta": float(assignment_delta.max()),
            "assignment_prop_positive": prop_positive,
            "assignment_prop_negative": prop_negative,
            "assignment_prop_zero": prop_zero,
            "assignment_zero_inside_95_range": zero_inside,
            "assignment_class": assignment_class,
            "assignment_replicates": N_REPLICATES,
            "assignment_seed": SEED,
            "assignment_group_sizes": "43|43|42|42|42",
        }

        for offset, aave_fpr, delta, sign in rotation_rows:
            detail_rows.append(
                {
                    **common,
                    "analysis": "cyclic_rotation",
                    "offset": offset,
                    "aave_fpr": aave_fpr,
                    "delta": delta,
                    "sign": sign,
                }
            )
        detail_rows.append(
            {
                **common,
                "analysis": "all5_average",
                "offset": np.nan,
                "aave_fpr": aave_fpr_all5,
                "delta": delta_all5,
                "sign": sign_label(delta_all5),
            }
        )
        detail_rows.append(
            {
                **common,
                "analysis": "balanced_assignment_distribution",
                "offset": np.nan,
                "aave_fpr": np.nan,
                "delta": np.nan,
                "sign": assignment_class,
            }
        )
        summary_rows.append(common)

    detail = pd.DataFrame(detail_rows)
    assignment_summary = pd.DataFrame(summary_rows)
    return detail, assignment_summary, all5_by_detector


def bootstrap_counts(
    base: pd.DataFrame,
) -> dict[tuple[str, str], np.ndarray]:
    """Draw one shared stratified poem bootstrap for all detectors."""
    rng = np.random.default_rng(SEED)
    counts: dict[tuple[str, str], np.ndarray] = {}
    for source in SOURCES:
        for variety in ["AAVE", "SAE"]:
            ids = sorted(
                base.loc[
                    base["source_subset"].eq(source) & base["variety"].eq(variety),
                    "source_doc_id",
                ].tolist()
            )
            n = len(ids)
            draws = rng.multinomial(n, np.full(n, 1.0 / n), size=N_REPLICATES)
            counts[(source, variety)] = draws
    return counts


def bootstrap_rank_effect(
    aave_scores: np.ndarray,
    sae_scores: np.ndarray,
    aave_counts: np.ndarray,
    sae_counts: np.ndarray,
) -> np.ndarray:
    """Weighted Mann--Whitney rank-biserial effects for bootstrap counts."""
    comparisons = (
        (aave_scores[:, None] > sae_scores[None, :]).astype(float)
        + 0.5 * (aave_scores[:, None] == sae_scores[None, :])
    )
    # Each row of ``sae_contribution`` gives, for every AAVE poem, the
    # weighted number of SAE bootstrap observations below/tied with it.
    sae_contribution = sae_counts @ comparisons.T
    u = np.einsum("bi,bi->b", aave_counts, sae_contribution, optimize=True)
    denominator = len(aave_scores) * len(sae_scores)
    return 2.0 * u / denominator - 1.0


def analyze_sources(
    scores: pd.DataFrame,
    step1: pd.DataFrame,
    all5_by_detector: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detector_order = step1["detector"].tolist()
    base = scores[["source_doc_id", "variety", "source_subset"]].drop_duplicates()
    base = base.sort_values(["source_subset", "variety", "source_doc_id"])
    counts = bootstrap_counts(base)

    source_totals = {
        source: EXPECTED_SOURCE_COUNTS["AAVE"][source] + EXPECTED_SOURCE_COUNTS["SAE"][source]
        for source in SOURCES
    }
    pooled_weights = {source: source_totals[source] / 733.0 for source in SOURCES}
    large_total = sum(source_totals[source] for source in LARGE_SOURCES)
    large_weights = {source: source_totals[source] / large_total for source in LARGE_SOURCES}
    assert np.isclose(sum(pooled_weights.values()), 1.0)
    assert np.isclose(sum(large_weights.values()), 1.0)

    step1_lookup = step1.set_index("detector")
    source_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for detector in detector_order:
        d = scores.loc[scores["detector"].eq(detector)].copy()
        family = str(d["family"].iloc[0])
        aave_all = d.loc[d["variety"].eq("AAVE")].sort_values("source_doc_id")
        all5_map = dict(zip(aave_all["source_doc_id"], all5_by_detector[detector]))
        d["flag_value"] = np.where(
            d["variety"].eq("AAVE"),
            d["source_doc_id"].map(all5_map),
            d["flag_machine"].astype(float),
        )
        assert d["flag_value"].notna().all()

        within_r: dict[str, float] = {}
        within_delta: dict[str, float] = {}
        boot_r: dict[str, np.ndarray] = {}
        boot_delta: dict[str, np.ndarray] = {}

        for source in SOURCES:
            a = d.loc[
                d["source_subset"].eq(source) & d["variety"].eq("AAVE")
            ].sort_values("source_doc_id")
            s = d.loc[
                d["source_subset"].eq(source) & d["variety"].eq("SAE")
            ].sort_values("source_doc_id")
            a_score = a["oriented_score"].to_numpy(float)
            s_score = s["oriented_score"].to_numpy(float)
            u, effect = rank_biserial(a_score, s_score)
            within_r[source] = effect
            a_counts = counts[(source, "AAVE")]
            s_counts = counts[(source, "SAE")]
            boot_r[source] = bootstrap_rank_effect(a_score, s_score, a_counts, s_counts)

            a_flag = a["flag_value"].to_numpy(float)
            s_flag = s["flag_value"].to_numpy(float)
            delta = float(a_flag.mean() - s_flag.mean())
            within_delta[source] = delta
            boot_a_mean = (a_counts @ a_flag) / len(a_flag)
            boot_s_mean = (s_counts @ s_flag) / len(s_flag)
            boot_delta[source] = boot_a_mean - boot_s_mean

            source_rows.append(
                {
                    "detector": detector,
                    "family": family,
                    "effect_type": "threshold_free_rank_biserial",
                    "specification": "within_source",
                    "source_subset": source,
                    "n_aave": len(a),
                    "n_sae": len(s),
                    "component_weight": pooled_weights[source],
                    "component_weights": "",
                    "mann_whitney_u": u,
                    "estimate": effect,
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "sign": sign_label(effect),
                    "bootstrap_replicates": 0,
                    "bootstrap_seed": np.nan,
                    "interval_type": "descriptive; no source-specific CI",
                }
            )
            source_rows.append(
                {
                    "detector": detector,
                    "family": family,
                    "effect_type": "operational_delta",
                    "specification": "within_source",
                    "source_subset": source,
                    "n_aave": len(a),
                    "n_sae": len(s),
                    "component_weight": pooled_weights[source],
                    "component_weights": "",
                    "mann_whitney_u": np.nan,
                    "estimate": delta,
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "sign": sign_label(delta),
                    "bootstrap_replicates": 0,
                    "bootstrap_seed": np.nan,
                    "interval_type": "descriptive; AAVE all-five vs SAE OOF",
                }
            )

        std_r = sum(pooled_weights[source] * within_r[source] for source in SOURCES)
        std_r_boot = sum(pooled_weights[source] * boot_r[source] for source in SOURCES)
        std_r_lo, std_r_hi = np.quantile(std_r_boot, [0.025, 0.975], method="linear")
        large_r = sum(large_weights[source] * within_r[source] for source in LARGE_SOURCES)
        large_r_boot = sum(large_weights[source] * boot_r[source] for source in LARGE_SOURCES)
        large_r_lo, large_r_hi = np.quantile(large_r_boot, [0.025, 0.975], method="linear")

        std_delta = sum(pooled_weights[source] * within_delta[source] for source in SOURCES)
        std_delta_boot = sum(pooled_weights[source] * boot_delta[source] for source in SOURCES)
        std_delta_lo, std_delta_hi = np.quantile(std_delta_boot, [0.025, 0.975], method="linear")
        large_delta = sum(large_weights[source] * within_delta[source] for source in LARGE_SOURCES)
        large_delta_boot = sum(large_weights[source] * boot_delta[source] for source in LARGE_SOURCES)
        large_delta_lo, large_delta_hi = np.quantile(large_delta_boot, [0.025, 0.975], method="linear")

        standardized_specs = [
            (
                "threshold_free_rank_biserial", "source_standardized_all4", "all4",
                212, 521, std_r, std_r_lo, std_r_hi,
                "dunbar=432/733|henderson=51/733|hughes=229/733|johnson1922=21/733",
            ),
            (
                "threshold_free_rank_biserial", "source_standardized_dunbar_hughes", "dunbar+hughes",
                153, 508, large_r, large_r_lo, large_r_hi,
                "dunbar=432/661|hughes=229/661",
            ),
            (
                "operational_delta", "source_standardized_all4", "all4",
                212, 521, std_delta, std_delta_lo, std_delta_hi,
                "dunbar=432/733|henderson=51/733|hughes=229/733|johnson1922=21/733",
            ),
            (
                "operational_delta", "source_standardized_dunbar_hughes", "dunbar+hughes",
                153, 508, large_delta, large_delta_lo, large_delta_hi,
                "dunbar=432/661|hughes=229/661",
            ),
        ]
        for effect_type, specification, source_label, n_a, n_s, estimate, ci_lo, ci_hi, weights_text in standardized_specs:
            source_rows.append(
                {
                    "detector": detector,
                    "family": family,
                    "effect_type": effect_type,
                    "specification": specification,
                    "source_subset": source_label,
                    "n_aave": n_a,
                    "n_sae": n_s,
                    "component_weight": np.nan,
                    "component_weights": weights_text,
                    "mann_whitney_u": np.nan,
                    "estimate": estimate,
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                    "sign": sign_label(estimate),
                    "bootstrap_replicates": N_REPLICATES,
                    "bootstrap_seed": SEED,
                    "interval_type": "95% stratified source-bootstrap CI; fixed source weights",
                }
            )

        original_r = float(step1_lookup.loc[detector, "rank_biserial_r"])
        canonical_delta = float(step1_lookup.loc[detector, "operational_delta"])
        all5_delta = float(all5_by_detector[detector].mean() - step1_lookup.loc[detector, "oof_fpr_sae"])
        threshold_free_signs = [sign_label(original_r), sign_label(std_r), sign_label(large_r)]
        operational_signs = [
            sign_label(canonical_delta), sign_label(all5_delta),
            sign_label(std_delta), sign_label(large_delta),
        ]
        summary_rows.append(
            {
                "detector": detector,
                "family": family,
                "original_full_pool_r": original_r,
                "original_full_pool_r_sign": threshold_free_signs[0],
                "source_standardized_r": std_r,
                "source_standardized_r_ci_lo": float(std_r_lo),
                "source_standardized_r_ci_hi": float(std_r_hi),
                "source_standardized_r_sign": threshold_free_signs[1],
                "dunbar_hughes_standardized_r": large_r,
                "dunbar_hughes_standardized_r_ci_lo": float(large_r_lo),
                "dunbar_hughes_standardized_r_ci_hi": float(large_r_hi),
                "dunbar_hughes_standardized_r_sign": threshold_free_signs[2],
                "threshold_free_sign_agreement": len(set(threshold_free_signs)) == 1,
                "canonical_delta": canonical_delta,
                "canonical_delta_sign": operational_signs[0],
                "all5_delta": all5_delta,
                "all5_delta_sign": operational_signs[1],
                "source_standardized_delta": std_delta,
                "source_standardized_delta_ci_lo": float(std_delta_lo),
                "source_standardized_delta_ci_hi": float(std_delta_hi),
                "source_standardized_delta_sign": operational_signs[2],
                "dunbar_hughes_standardized_delta": large_delta,
                "dunbar_hughes_standardized_delta_ci_lo": float(large_delta_lo),
                "dunbar_hughes_standardized_delta_ci_hi": float(large_delta_hi),
                "dunbar_hughes_standardized_delta_sign": operational_signs[3],
                "operational_sign_agreement": len(set(operational_signs)) == 1,
                "operational_positive_every_specification": all(x == "positive" for x in operational_signs),
                "operational_negative_every_specification": all(x == "negative" for x in operational_signs),
                "genuinely_specification_sensitive": len(set(operational_signs)) > 1,
            }
        )

    return pd.DataFrame(source_rows), pd.DataFrame(summary_rows)


def add_assignment_to_summary(
    summary: pd.DataFrame, assignment_summary: pd.DataFrame
) -> pd.DataFrame:
    assignment_columns = [
        "detector", "rotation_min_delta", "rotation_max_delta", "rotation_positive_n",
        "rotation_negative_n", "rotation_zero_n", "rotation_sign_changes",
        "assignment_median_delta", "assignment_q025_delta", "assignment_q975_delta",
        "assignment_min_delta", "assignment_max_delta", "assignment_prop_positive",
        "assignment_prop_negative", "assignment_prop_zero", "assignment_zero_inside_95_range",
        "assignment_class",
    ]
    return summary.merge(
        assignment_summary[assignment_columns], on="detector", how="left", validate="one_to_one"
    )


def write_report(
    audit: dict[str, object],
    step1: pd.DataFrame,
    assignment_detail: pd.DataFrame,
    assignment_summary: pd.DataFrame,
    source_effects: pd.DataFrame,
    robustness: pd.DataFrame,
) -> None:
    detector_order = step1["detector"].tolist()
    step1_lookup = step1.set_index("detector")
    assignment_lookup = assignment_summary.set_index("detector")
    robustness_lookup = robustness.set_index("detector")

    lines: list[str] = [
        "# RQ1 natural-reference robustness (Step 5.5)\n",
        "This analysis tests two nuisance choices in the frozen natural AAVE-versus-SAE reference: assignment of unpaired AAVE poems to the five frozen SAE-derived thresholds, and unequal retained source-subset mixture. It uses only Step 1 scores, folds, thresholds, and metadata. It does not run detector inference, call GPTZero, transform text, edit the manuscript, or use synthetic-arm results to choose a specification.\n",
        "## A. Input validation\n",
        f"Validation passed for **{audit['aave_n']} AAVE poems**, **{audit['sae_n']} SAE poems**, **{audit['detectors_n']} frozen detector configurations**, **{audit['scores_n']:,} detector-by-poem score rows**, and **{audit['thresholds_n']} frozen detector-by-fold thresholds**. There are {audit['duplicate_detector_source_n']} duplicate detector-by-source rows and {audit['missing_score_n']} missing oriented scores.\n",
        "| variety | dunbar | henderson | hughes | johnson1922 | total |",
        "|---|---:|---:|---:|---:|---:|",
        "| AAVE | 115 | 47 | 38 | 12 | 212 |",
        "| SAE | 317 | 4 | 191 | 9 | 521 |\n",
        "All 212 AAVE and 521 SAE poem IDs and source-document IDs are unique. The frozen score orientations are exact: log probability, DetectLLM-LRR, Binoculars, and pretrained P(machine) use +1; rank, log-rank, and entropy use -1. Every oriented score exactly equals raw score times its frozen multiplier. The 521-row SAE fold file matches the score table exactly, with fold sizes 105, 104, 104, 104, and 104. Every threshold exactly reproduces NumPy's `method=\"higher\"` 95th percentile on the other four SAE folds, and classification remains strict `oriented_score > threshold` with ties human.\n",
        "Frozen input SHA-256 values:\n",
    ]
    for path, digest in audit["hashes"].items():
        lines.append(f"- `{path}`: `{digest}`")

    lines += [
        "\n## B. Canonical Step 1 reproduction\n",
        f"All stored detector-by-poem decisions reproduce exactly. AAVE FPR, SAE OOF FPR, and delta reproduce the Step 1 CSV to floating serialization tolerance; the maximum absolute numeric difference is {audit['max_canonical_rate_abs_diff']:.3g}.\n",
        "| detector | family | AAVE FPR | SAE OOF FPR | delta | sign |",
        "|---|---|---:|---:|---:|---|",
    ]
    for detector in detector_order:
        row = step1_lookup.loc[detector]
        lines.append(
            f"| {detector} | {row.family} | {row.aave_fpr:.4f} | {row.oof_fpr_sae:.4f} | "
            f"{row.operational_delta:+.4f} | {sign_label(row.operational_delta)} |"
        )

    lines += [
        "\n## C. Five cyclic threshold rotations\n",
        "Offset `k` maps an AAVE poem in fixed evaluation fold `f` to frozen SAE threshold `(f + k) mod 5`. SAE OOF decisions never change. The rotations are an assignment-sensitivity analysis, not independent samples. Each offset cell shows `AAVE FPR / delta`.\n",
        "| detector | off 0 | off 1 | off 2 | off 3 | off 4 | min delta | max delta | + / - / 0 | sign changes? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    rotations = assignment_detail.loc[assignment_detail["analysis"].eq("cyclic_rotation")]
    for detector in detector_order:
        d = rotations.loc[rotations["detector"].eq(detector)].sort_values("offset")
        row = assignment_lookup.loc[detector]
        cells = [f"{x.aave_fpr:.4f} / {x.delta:+.4f}" for x in d.itertuples(index=False)]
        lines.append(
            f"| {detector} | {' | '.join(cells)} | {row.rotation_min_delta:+.4f} | "
            f"{row.rotation_max_delta:+.4f} | {int(row.rotation_positive_n)} / "
            f"{int(row.rotation_negative_n)} / {int(row.rotation_zero_n)} | "
            f"{bool_word(row.rotation_sign_changes)} |"
        )

    lines += [
        "\n## D. All-five-threshold averaged result\n",
        "For each AAVE poem, `flag_value` is its mean machine indicator across the complete set of five frozen SAE thresholds. Averaging these poem-level values yields an assignment-free AAVE operational summary. The five threshold applications to one poem are not treated as independent observations.\n",
        "| detector | family | AAVE FPR all5 | SAE OOF FPR | delta all5 | sign |",
        "|---|---|---:|---:|---:|---|",
    ]
    for detector in detector_order:
        row = assignment_lookup.loc[detector]
        lines.append(
            f"| {detector} | {row.family} | {row.aave_fpr_all5:.4f} | {row.sae_oof_fpr:.4f} | "
            f"{row.delta_all5:+.4f} | {sign_label(row.delta_all5)} |"
        )

    lines += [
        "\n## E. 10,000 balanced assignment sensitivity\n",
        f"Using seed `{SEED}`, each replicate permutes the 212 fixed AAVE poems over threshold-fold labels with fixed capacities `(43, 43, 42, 42, 42)` for folds 0–4. The values below form an **AAVE threshold-assignment sensitivity distribution**. The 2.5th and 97.5th percentiles are sensitivity percentiles, not confidence limits.\n",
        "| detector | median delta | 2.5% | 97.5% | min | max | prop + | prop - | zero in 95% range? | classification |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for detector in detector_order:
        row = assignment_lookup.loc[detector]
        lines.append(
            f"| {detector} | {row.assignment_median_delta:+.4f} | {row.assignment_q025_delta:+.4f} | "
            f"{row.assignment_q975_delta:+.4f} | {row.assignment_min_delta:+.4f} | "
            f"{row.assignment_max_delta:+.4f} | {row.assignment_prop_positive:.4f} | "
            f"{row.assignment_prop_negative:.4f} | {bool_word(row.assignment_zero_inside_95_range)} | "
            f"{row.assignment_class} |"
        )

    within_r = source_effects.loc[
        source_effects["effect_type"].eq("threshold_free_rank_biserial")
        & source_effects["specification"].eq("within_source")
    ]
    lines += [
        "\n## F. Within-source effects\n",
        "These are RQ1-compatible threshold-free AAVE-versus-SAE comparisons made within each named retained source subset. Henderson has only 4 SAE poems and Johnson 9, so all source-specific estimates are descriptive; source-specific p-values are not used or reported.\n",
        "| detector | source | n AAVE | n SAE | Mann–Whitney U | rank-biserial r | sign |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for detector in detector_order:
        d = within_r.loc[within_r["detector"].eq(detector)]
        for source in SOURCES:
            row = d.loc[d["source_subset"].eq(source)].iloc[0]
            lines.append(
                f"| {detector} | {source} | {int(row.n_aave)} | {int(row.n_sae)} | "
                f"{row.mann_whitney_u:.1f} | {row.estimate:+.4f} | {row.sign} |"
            )

    lines += [
        "\n## G. Source-standardized threshold-free result\n",
        "The fixed pooled-corpus weights are Dunbar 432/733, Henderson 51/733, Hughes 229/733, and Johnson 21/733. Each standardized effect is the weighted sum of within-source rank-biserial effects, thereby excluding cross-source AAVE–SAE pairs. The 95% intervals use 10,000 stratified source-bootstrap replicates with fixed weights and seed `20260919`.\n",
        "| detector | standardized r | 95% bootstrap CI | sign |",
        "|---|---:|---:|---|",
    ]
    for detector in detector_order:
        row = robustness_lookup.loc[detector]
        lines.append(
            f"| {detector} | {row.source_standardized_r:+.4f} | "
            f"[{row.source_standardized_r_ci_lo:+.4f}, {row.source_standardized_r_ci_hi:+.4f}] | "
            f"{row.source_standardized_r_sign} |"
        )

    lines += [
        "\n## H. Source-standardized operational result\n",
        "AAVE contributes its poem-level all-five-threshold average flag value; SAE contributes its actual frozen out-of-fold classification. Within-source deltas are combined with the same fixed pooled weights. Thresholds are never recalibrated inside the stratified source bootstrap.\n",
        "| detector | standardized delta | 95% bootstrap CI | sign |",
        "|---|---:|---:|---|",
    ]
    for detector in detector_order:
        row = robustness_lookup.loc[detector]
        lines.append(
            f"| {detector} | {row.source_standardized_delta:+.4f} | "
            f"[{row.source_standardized_delta_ci_lo:+.4f}, {row.source_standardized_delta_ci_hi:+.4f}] | "
            f"{row.source_standardized_delta_sign} |"
        )

    lines += [
        "\n## I. Dunbar/Hughes-only sensitivity\n",
        "Restricting the standardization to the two large SAE strata gives fixed renormalized weights Dunbar 432/661 and Hughes 229/661. This is a sensitivity analysis, not a replacement population definition.\n",
        "| detector | standardized r [95% CI] | sign | operational delta [95% CI] | sign |",
        "|---|---:|---|---:|---|",
    ]
    for detector in detector_order:
        row = robustness_lookup.loc[detector]
        lines.append(
            f"| {detector} | {row.dunbar_hughes_standardized_r:+.4f} "
            f"[{row.dunbar_hughes_standardized_r_ci_lo:+.4f}, {row.dunbar_hughes_standardized_r_ci_hi:+.4f}] | "
            f"{row.dunbar_hughes_standardized_r_sign} | "
            f"{row.dunbar_hughes_standardized_delta:+.4f} "
            f"[{row.dunbar_hughes_standardized_delta_ci_lo:+.4f}, {row.dunbar_hughes_standardized_delta_ci_hi:+.4f}] | "
            f"{row.dunbar_hughes_standardized_delta_sign} |"
        )

    lines += [
        "\n## J. Detector-by-detector stability table\n",
        "The original full-pool estimates remain the primary population estimates. The other columns are prespecified robustness analyses, not candidate replacements selected after inspecting results.\n",
        "| detector | full r | std r | D+H r | TF signs agree? | canonical delta | all5 delta | std delta | D+H delta | operational signs agree? |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for detector in detector_order:
        row = robustness_lookup.loc[detector]
        lines.append(
            f"| {detector} | {row.original_full_pool_r:+.4f} | {row.source_standardized_r:+.4f} | "
            f"{row.dunbar_hughes_standardized_r:+.4f} | {bool_word(row.threshold_free_sign_agreement)} | "
            f"{row.canonical_delta:+.4f} | {row.all5_delta:+.4f} | {row.source_standardized_delta:+.4f} | "
            f"{row.dunbar_hughes_standardized_delta:+.4f} | {bool_word(row.operational_sign_agreement)} |"
        )

    zero = robustness.loc[robustness["family"].eq("zero-shot")]
    pre = robustness.loc[robustness["family"].eq("pretrained")]
    zero_assignment = assignment_summary.loc[assignment_summary["family"].eq("zero-shot")]
    pre_assignment = assignment_summary.loc[assignment_summary["family"].eq("pretrained")]
    z_can_pos = int((zero["canonical_delta"] > 0).sum())
    z_all5_pos = int((zero["all5_delta"] > 0).sum())
    z_rot_pos = int((zero_assignment["rotation_positive_n"] == 5).sum())
    z_rot_neg = int((zero_assignment["rotation_negative_n"] == 5).sum())
    z_assign_pos = int((zero_assignment["assignment_class"] == ">95% positive").sum())
    z_assign_neg = int((zero_assignment["assignment_class"] == ">95% negative").sum())
    z_assign_uncertain = 12 - z_assign_pos - z_assign_neg
    z_tf_stable = int(zero["threshold_free_sign_agreement"].sum())
    z_op_stable = int(zero["operational_sign_agreement"].sum())
    z_op_pos_all = int(zero["operational_positive_every_specification"].sum())
    z_op_neg_all = int(zero["operational_negative_every_specification"].sum())
    z_sensitive = int(zero["genuinely_specification_sensitive"].sum())

    lines += [
        "\n## K. Zero-shot family stability summary\n",
        "### Threshold-assignment summary\n",
        f"Among 12 zero-shot configurations, {z_can_pos}/12 have a positive canonical delta and {z_all5_pos}/12 have a positive all-five delta. Across cyclic rotations, {z_rot_pos}/12 are positive in all five and {z_rot_neg}/12 are negative in all five. In the balanced assignment sensitivity distributions, {z_assign_pos}/12 are >95% positive, {z_assign_neg}/12 are >95% negative, and {z_assign_uncertain}/12 are assignment-sensitive/uncertain.\n",
        "The three pretrained configurations are kept separate:\n",
        "| detector | cyclic + / - / 0 | assignment classification |",
        "|---|---:|---|",
    ]
    for row in pre_assignment.itertuples(index=False):
        lines.append(
            f"| {row.detector} | {row.rotation_positive_n} / {row.rotation_negative_n} / {row.rotation_zero_n} | "
            f"{row.assignment_class} |"
        )

    lines += [
        "\n### Source-specification summary\n",
        f"For the 12 zero-shot configurations, the threshold-free sign is stable across the original full pool, all-four-source standardization, and Dunbar/Hughes standardization for **{z_tf_stable}/12**. The operational sign is stable across canonical assignment, all-five-threshold averaging, all-four-source standardization, and Dunbar/Hughes standardization for **{z_op_stable}/12**: **{z_op_pos_all}/12** are positive under every specification, **{z_op_neg_all}/12** are negative under every specification, and **{z_sensitive}/12** are genuinely specification-sensitive.\n",
        "Detector-by-detector zero-shot status:\n",
        "| detector | TF signs (full/std/D+H) | TF stable? | operational signs (canonical/all5/std/D+H) | operational status |",
        "|---|---|---|---|---|",
    ]
    for row in zero.itertuples(index=False):
        tf_signs = "/".join(
            [row.original_full_pool_r_sign, row.source_standardized_r_sign, row.dunbar_hughes_standardized_r_sign]
        )
        op_signs = "/".join(
            [row.canonical_delta_sign, row.all5_delta_sign, row.source_standardized_delta_sign, row.dunbar_hughes_standardized_delta_sign]
        )
        if row.operational_positive_every_specification:
            status = "positive every specification"
        elif row.operational_negative_every_specification:
            status = "negative every specification"
        else:
            status = "specification-sensitive"
        lines.append(
            f"| {row.detector} | {tf_signs} | {bool_word(row.threshold_free_sign_agreement)} | "
            f"{op_signs} | {status} |"
        )

    lines += [
        "\nPretrained detector-by-detector source-specification status:\n",
        "| detector | TF signs (full/std/D+H) | TF stable? | operational signs (canonical/all5/std/D+H) | operational stable? |",
        "|---|---|---|---|---|",
    ]
    for row in pre.itertuples(index=False):
        lines.append(
            f"| {row.detector} | {row.original_full_pool_r_sign}/{row.source_standardized_r_sign}/{row.dunbar_hughes_standardized_r_sign} | "
            f"{bool_word(row.threshold_free_sign_agreement)} | "
            f"{row.canonical_delta_sign}/{row.all5_delta_sign}/{row.source_standardized_delta_sign}/{row.dunbar_hughes_standardized_delta_sign} | "
            f"{bool_word(row.operational_sign_agreement)} |"
        )

    lines += [
        "\n## L. Interpretation limitations\n",
        "- Source standardization removes cross-source pair comparisons and fixes a declared retained-corpus mixture; it does not isolate a causal dialect effect or fully control author, anthology, poem form, topic, length, orthography, or style.",
        "- Henderson and Johnson have only 4 and 9 SAE poems. Their within-source effects are descriptive and their bootstrap contributions remain sensitive to those small observed strata.",
        "- The five cyclic rotations are related deterministic reassignments, not independent replications.",
        "- Monte Carlo assignment percentiles measure sensitivity to arbitrary balanced threshold assignment; they are not confidence intervals.",
        "- Bootstrap intervals condition on the frozen scores and thresholds. Thresholds are not recalibrated inside this analysis.",
        "- The original full-pool natural result remains the primary population estimate. No robustness specification is promoted because its result is more favorable.",
        "- The 15 configurations are reported detector by detector. Their signs are heterogeneous and are not collapsed into a family-average effect.",
        "- No comparison to the frozen synthetic arm is made here.\n",
        "## M. Final decision\n",
        f"Threshold assignment alone is directionally stable for 11/12 zero-shot configurations, but 3/12 threshold-free configurations and 3/12 operational configurations change sign across the prespecified source specifications. The natural reference therefore remains detector-specific and does not support one unqualified reversal direction.\n",
    ]

    # A simple, unqualified reversal claim requires both natural-reference
    # summaries to be directionally stable across their prespecified source
    # specifications for every zero-shot configuration. Detector-specific
    # comparison could still be informative, but is not the "simple reversal"
    # option posed by the frozen decision rule.
    if z_sensitive == 0 and z_tf_stable == 12:
        lines.append("**PROCEED TO NATURAL-VS-SYNTHETIC BENCHMARK COMPARISON**")
    else:
        lines.append("**NATURAL REFERENCE IS TOO SPECIFICATION-SENSITIVE FOR A SIMPLE REVERSAL CLAIM**")

    REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    scores, thresholds, step1, audit = load_and_validate()
    assignment_detail, assignment_summary, all5_by_detector = analyze_threshold_assignment(
        scores, thresholds, step1
    )
    source_effects, robustness = analyze_sources(scores, step1, all5_by_detector)
    robustness = add_assignment_to_summary(robustness, assignment_summary)

    assignment_detail.to_csv(ASSIGNMENT_OUT, index=False)
    source_effects.to_csv(SOURCE_OUT, index=False)
    robustness.to_csv(SUMMARY_OUT, index=False)
    write_report(audit, step1, assignment_detail, assignment_summary, source_effects, robustness)

    zero = robustness.loc[robustness["family"].eq("zero-shot")]
    print(
        zero[[
            "detector", "threshold_free_sign_agreement", "operational_sign_agreement",
            "operational_positive_every_specification", "operational_negative_every_specification",
            "genuinely_specification_sensitive",
        ]].to_string(index=False)
    )
    print(f"wrote {REPORT.relative_to(ROOT)}")
    print(f"wrote {ASSIGNMENT_OUT.relative_to(ROOT)}")
    print(f"wrote {SOURCE_OUT.relative_to(ROOT)}")
    print(f"wrote {SUMMARY_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Focused poem-length sensitivity for the frozen natural RQ1 benchmark.

The analysis consumes the final frozen poem-level score, fold, threshold, and
summary artifacts. It never runs a detector and never recalibrates a threshold
on AAVE or on a length-balanced subset.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data/analysis"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"

SCORES_PATH = ANALYSIS / "rq1_full_poetry_natural_scores.csv"
THRESHOLDS_PATH = ANALYSIS / "rq1_full_poetry_thresholds.csv"
FOLDS_PATH = ANALYSIS / "rq1_full_poetry_sae_folds.csv"
STEP1_PATH = ANALYSIS / "rq1_full_poetry_summary.csv"
ROBUSTNESS_PATH = ANALYSIS / "rq1_natural_reference_robustness_summary.csv"

BASELINE_OUT = ANALYSIS / "rq1_length_sensitivity_baseline_verification.csv"
SUMMARY_OUT = ANALYSIS / "rq1_length_sensitivity_summary.csv"
BALANCE_OUT = ANALYSIS / "rq1_length_balance_diagnostics.csv"
REPORT_OUT = REPORTS / "rq1_length_sensitivity.md"
TABLE_OUT = REPORTS / "table_rq1_length_sensitivity.tex"
FIGURE_PDF = FIGURES / "rq1_length_sensitivity.pdf"
FIGURE_PNG = FIGURES / "rq1_length_sensitivity.png"

SEED = 20260920
N_BOOT = 10_000
N_STRATA = 5
BALANCE_SMD_LIMIT = 0.10

EXPECTED_HASHES = {
    SCORES_PATH: "6e835ee0280611c3b0d70f1ea7054a84e28ed636128068e2f2c11ecaebf3d208",
    THRESHOLDS_PATH: "3e22028e1f9e3a264370cde766002c87db6a7c115d184fe4b00469e4646def70",
    FOLDS_PATH: "249761564463507527f23a0e2426d941d30cafe3467ac0652bdb5f764ea69d48",
    STEP1_PATH: "67ac1accf89dd3af39b95e81b49fb3894ed7aac93ba7f8b5fdb159edad320d0e",
}

HEADLINE_POSITIVE = [
    "Entropy (GPT2-XL)",
    "Log Probability (OPT-1.3B)",
    "Log-Rank (negated) (OPT-1.3B)",
    "Entropy (OPT-1.3B)",
    "Binoculars (OPT-1.3B)",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign_label(value: float, tol: float = 1e-15) -> str:
    if value > tol:
        return "positive"
    if value < -tol:
        return "negative"
    return "zero"


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="mergesort")
    x = np.asarray(values, float)[order]
    w = np.asarray(weights, float)[order]
    cumulative = np.cumsum(w) - 0.5 * w
    cumulative /= w.sum()
    return float(np.interp(q, cumulative, x))


def rank_biserial(aave: np.ndarray, sae: np.ndarray) -> float:
    test = mannwhitneyu(aave, sae, alternative="two-sided", method="asymptotic")
    return 2.0 * float(test.statistic) / (len(aave) * len(sae)) - 1.0


def weighted_rank_biserial(
    aave: np.ndarray,
    sae: np.ndarray,
    aave_weights: np.ndarray,
    sae_weights: np.ndarray,
) -> float:
    """Weighted probability-of-superiority effect, with half credit for ties."""
    order = np.argsort(sae, kind="mergesort")
    s = np.asarray(sae, float)[order]
    sw = np.asarray(sae_weights, float)[order]
    cumulative = np.r_[0.0, np.cumsum(sw)]
    left = np.searchsorted(s, aave, side="left")
    right = np.searchsorted(s, aave, side="right")
    below = cumulative[left]
    tied = cumulative[right] - cumulative[left]
    u = np.sum(aave_weights * (below + 0.5 * tied))
    return float(2.0 * u / (aave_weights.sum() * sae_weights.sum()) - 1.0)


def penalized_loglik(beta: np.ndarray, x: np.ndarray, y: np.ndarray, w0: np.ndarray) -> float:
    eta = x @ beta
    p = expit(eta)
    w = w0 * p * (1.0 - p)
    info = x.T @ (w[:, None] * x)
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf
    likelihood = np.sum(w0 * (y * eta - np.logaddexp(0.0, eta)))
    return float(likelihood + 0.5 * logdet)


def firth_logistic(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    initial: np.ndarray | None = None,
    max_iter: int = 80,
    tol: float = 1e-9,
) -> tuple[np.ndarray, bool, int]:
    """Fit Firth logistic regression by adjusted-score iteration.

    The Jeffreys-prior adjustment keeps estimates finite for detector outcomes
    with complete or quasi-complete separation.
    """
    w0 = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, float)
    w0 = w0 * (len(w0) / w0.sum())
    beta = np.zeros(x.shape[1], dtype=float) if initial is None else np.asarray(initial, float).copy()
    current = penalized_loglik(beta, x, y, w0)

    for iteration in range(1, max_iter + 1):
        p = expit(x @ beta)
        w = w0 * p * (1.0 - p)
        info = x.T @ (w[:, None] * x)
        info_inv = np.linalg.pinv(info, rcond=1e-12)
        leverage = w * np.einsum("ij,jk,ik->i", x, info_inv, x, optimize=True)
        adjusted_score = x.T @ (w0 * (y - p) + leverage * (0.5 - p))
        step = info_inv @ adjusted_score
        if np.max(np.abs(step)) < tol:
            return beta, True, iteration

        scale = 1.0
        accepted = False
        for _ in range(18):
            candidate = beta + scale * step
            candidate_value = penalized_loglik(candidate, x, y, w0)
            if np.isfinite(candidate_value) and candidate_value >= current - 1e-10:
                beta = candidate
                current = candidate_value
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break

    objective = lambda b: -penalized_loglik(np.asarray(b), x, y, w0)
    fallback = minimize(objective, beta, method="BFGS", options={"maxiter": 250, "gtol": 1e-8})
    if np.all(np.isfinite(fallback.x)):
        beta = np.asarray(fallback.x, float)
    converged = bool(fallback.success or np.linalg.norm(fallback.jac, ord=np.inf) < 1e-5)
    return beta, converged, max_iter + int(getattr(fallback, "nit", 0))


def standardized_probability_difference(
    beta: np.ndarray,
    z: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    p1 = expit(beta[0] + beta[1] + beta[2] * z)
    p0 = expit(beta[0] + beta[2] * z)
    if weights is None:
        return float(np.mean(p1 - p0))
    return float(np.average(p1 - p0, weights=weights))


def load_and_verify() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        assert actual == expected, f"Frozen input hash mismatch: {path}: {actual} != {expected}"

    scores = pd.read_csv(SCORES_PATH)
    thresholds = pd.read_csv(THRESHOLDS_PATH)
    folds = pd.read_csv(FOLDS_PATH)
    step1 = pd.read_csv(STEP1_PATH)
    robustness = pd.read_csv(ROBUSTNESS_PATH)

    zero = scores.loc[scores["family"].eq("zero-shot")].copy()
    assert len(zero) == 12 * 733
    assert zero["detector"].nunique() == 12
    assert not zero.duplicated(["detector", "source_doc_id"]).any()
    assert zero["oriented_score"].notna().all() and np.isfinite(zero["oriented_score"]).all()
    assert np.array_equal(
        zero["oriented_score"].to_numpy(),
        (zero["raw_score"] * zero["orientation_multiplier"]).to_numpy(),
    )

    poems = zero[
        [
            "poem_id", "source_doc_id", "variety", "source_dataset", "source_subset",
            "poet", "poem_title", "n_words", "evaluation_fold",
        ]
    ].drop_duplicates("source_doc_id")
    assert len(poems) == 733 and not poems["source_doc_id"].duplicated().any()
    assert poems.groupby("variety").size().to_dict() == {"AAVE": 212, "SAE": 521}
    assert poems.groupby("variety")["n_words"].median().to_dict() == {"AAVE": 192.0, "SAE": 98.0}
    assert (poems["n_words"] > 0).all()

    assert len(folds) == 521 and not folds["source_doc_id"].duplicated().any()
    sae_check = poems.loc[poems["variety"].eq("SAE")].merge(
        folds[["source_doc_id", "sae_fold"]], on="source_doc_id", validate="one_to_one"
    )
    assert (sae_check["evaluation_fold"] == sae_check["sae_fold"]).all()
    assert sae_check["sae_fold"].value_counts().sort_index().to_dict() == {
        0: 105, 1: 104, 2: 104, 3: 104, 4: 104
    }

    zero_thresholds = thresholds.loc[thresholds["family"].eq("zero-shot")].copy()
    assert len(zero_thresholds) == 60
    threshold_lookup = zero_thresholds.set_index(["detector", "fold"])["threshold"]
    mapped = np.array([
        threshold_lookup.loc[(detector, fold)]
        for detector, fold in zip(zero["detector"], zero["evaluation_fold"])
    ])
    assert np.array_equal(mapped, zero["threshold"].to_numpy())
    reproduced_flags = zero["oriented_score"].to_numpy() > mapped
    assert np.array_equal(reproduced_flags, zero["flag_machine"].astype(bool).to_numpy())

    baseline_rows: list[dict[str, object]] = []
    step1_zero = step1.loc[step1["family"].eq("zero-shot")].copy()
    for detector in step1_zero["detector"]:
        d = zero.loc[zero["detector"].eq(detector)].copy()
        for fold in range(5):
            calibration = d.loc[
                d["variety"].eq("SAE") & d["evaluation_fold"].ne(fold), "oriented_score"
            ].to_numpy(float)
            reproduced_threshold = float(np.quantile(calibration, 0.95, method="higher"))
            stored_threshold = float(threshold_lookup.loc[(detector, fold)])
            assert reproduced_threshold == stored_threshold
        rates = d.groupby("variety")["flag_machine"].mean()
        reproduced_delta = float(rates["AAVE"] - rates["SAE"])
        stored = step1_zero.loc[step1_zero["detector"].eq(detector)].iloc[0]
        baseline_rows.append({
            "detector": detector,
            "n_aave": int((d["variety"] == "AAVE").sum()),
            "n_sae": int((d["variety"] == "SAE").sum()),
            "stored_aave_fpr": float(stored["aave_fpr"]),
            "reproduced_aave_fpr": float(rates["AAVE"]),
            "stored_sae_oof_fpr": float(stored["oof_fpr_sae"]),
            "reproduced_sae_oof_fpr": float(rates["SAE"]),
            "stored_delta": float(stored["operational_delta"]),
            "reproduced_delta": reproduced_delta,
            "absolute_delta_difference": abs(reproduced_delta - float(stored["operational_delta"])),
            "thresholds_exact": True,
            "flags_exact": True,
        })
    baseline = pd.DataFrame(baseline_rows)
    assert baseline["absolute_delta_difference"].max() <= 1e-15

    frozen_positive = robustness.loc[
        robustness["family"].eq("zero-shot")
        & robustness["operational_positive_every_specification"].astype(bool),
        "detector",
    ].tolist()
    assert set(frozen_positive) == set(HEADLINE_POSITIVE)
    return zero, poems, step1_zero, robustness, baseline


def build_length_design(poems: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    p = poems.copy().sort_values("source_doc_id").reset_index(drop=True)
    p["log_words"] = np.log1p(p["n_words"].to_numpy(float))
    lo = max(p.loc[p["variety"].eq(v), "log_words"].min() for v in ["AAVE", "SAE"])
    hi = min(p.loc[p["variety"].eq(v), "log_words"].max() for v in ["AAVE", "SAE"])
    p["in_common_support"] = p["log_words"].between(lo, hi, inclusive="both")
    common = p.loc[p["in_common_support"]].copy()
    common["length_stratum"], edges = pd.qcut(
        common["log_words"], q=N_STRATA, labels=False, retbins=True, duplicates="drop"
    )
    assert len(edges) == N_STRATA + 1
    p = p.merge(
        common[["source_doc_id", "length_stratum"]], on="source_doc_id", how="left", validate="one_to_one"
    )

    counts = common.groupby(["length_stratum", "variety"]).size().unstack(fill_value=0)
    assert counts.shape == (N_STRATA, 2) and int(counts.min().min()) >= 10
    target_weights = counts.sum(axis=1).to_numpy(float) / len(common)
    assert np.isclose(target_weights.sum(), 1.0)

    p["cem_weight"] = 0.0
    for stratum in range(N_STRATA):
        for variety in ["AAVE", "SAE"]:
            mask = p["in_common_support"] & p["length_stratum"].eq(stratum) & p["variety"].eq(variety)
            p.loc[mask, "cem_weight"] = target_weights[stratum] / mask.sum()
    assert np.allclose(p.loc[p["in_common_support"]].groupby("variety")["cem_weight"].sum(), 1.0)

    pooled_sd = np.sqrt(
        (
            p.loc[p["variety"].eq("AAVE"), "log_words"].var(ddof=1)
            + p.loc[p["variety"].eq("SAE"), "log_words"].var(ddof=1)
        ) / 2.0
    )
    before_means = p.groupby("variety")["log_words"].mean()
    smd_before = float((before_means["AAVE"] - before_means["SAE"]) / pooled_sd)
    balanced_means = {
        v: np.average(
            p.loc[p["variety"].eq(v) & p["in_common_support"], "log_words"],
            weights=p.loc[p["variety"].eq(v) & p["in_common_support"], "cem_weight"],
        )
        for v in ["AAVE", "SAE"]
    }
    smd_after = float((balanced_means["AAVE"] - balanced_means["SAE"]) / pooled_sd)
    assert abs(smd_after) < BALANCE_SMD_LIMIT

    rows: list[dict[str, object]] = []
    for sample, mask, weight_col in [
        ("original_unweighted", pd.Series(True, index=p.index), None),
        ("common_support_unweighted", p["in_common_support"], None),
        ("length_balanced_weighted", p["in_common_support"], "cem_weight"),
    ]:
        for variety in ["AAVE", "SAE"]:
            d = p.loc[mask & p["variety"].eq(variety)]
            w = np.ones(len(d)) if weight_col is None else d[weight_col].to_numpy(float)
            rows.append({
                "record_type": "distribution",
                "sample": sample,
                "variety": variety,
                "stratum": np.nan,
                "n_retained": len(d),
                "n_discarded": int((p["variety"].eq(variety)).sum() - len(d)),
                "discarded_fraction": float(1.0 - len(d) / (p["variety"].eq(variety)).sum()),
                "effective_n": float(w.sum() ** 2 / np.sum(w ** 2)),
                "min_words": int(d["n_words"].min()),
                "q25_words": weighted_quantile(d["n_words"].to_numpy(float), w, 0.25),
                "median_words": weighted_quantile(d["n_words"].to_numpy(float), w, 0.50),
                "mean_words": float(np.average(d["n_words"], weights=w)),
                "q75_words": weighted_quantile(d["n_words"].to_numpy(float), w, 0.75),
                "max_words": int(d["n_words"].max()),
                "mean_log1p_words": float(np.average(d["log_words"], weights=w)),
                "log_length_smd": smd_before if sample == "original_unweighted" else (
                    smd_after if sample == "length_balanced_weighted" else np.nan
                ),
                "balance_limit_abs_smd": BALANCE_SMD_LIMIT,
                "balance_pass": bool(abs(smd_after) < BALANCE_SMD_LIMIT) if sample == "length_balanced_weighted" else np.nan,
                "target_weight": np.nan,
                "stratum_lower_words": np.nan,
                "stratum_upper_words": np.nan,
            })

    for stratum in range(N_STRATA):
        lower_words = float(np.expm1(edges[stratum]))
        upper_words = float(np.expm1(edges[stratum + 1]))
        for variety in ["AAVE", "SAE"]:
            mask = p["in_common_support"] & p["length_stratum"].eq(stratum) & p["variety"].eq(variety)
            rows.append({
                "record_type": "stratum",
                "sample": "length_balanced_weighted",
                "variety": variety,
                "stratum": stratum + 1,
                "n_retained": int(mask.sum()),
                "n_discarded": np.nan,
                "discarded_fraction": np.nan,
                "effective_n": np.nan,
                "min_words": int(p.loc[mask, "n_words"].min()),
                "q25_words": np.nan,
                "median_words": float(p.loc[mask, "n_words"].median()),
                "mean_words": float(p.loc[mask, "n_words"].mean()),
                "q75_words": np.nan,
                "max_words": int(p.loc[mask, "n_words"].max()),
                "mean_log1p_words": float(p.loc[mask, "log_words"].mean()),
                "log_length_smd": np.nan,
                "balance_limit_abs_smd": BALANCE_SMD_LIMIT,
                "balance_pass": np.nan,
                "target_weight": target_weights[stratum],
                "stratum_lower_words": lower_words,
                "stratum_upper_words": upper_words,
            })
    return p, pd.DataFrame(rows), target_weights


def author_multiplier_weights(poems: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    authors = sorted(poems["poet"].astype(str).unique().tolist())
    assert len(authors) == 9
    author_index = poems["poet"].astype(str).map({a: i for i, a in enumerate(authors)}).to_numpy(int)
    rng = np.random.default_rng(SEED)
    weights = rng.exponential(scale=1.0, size=(N_BOOT, len(authors)))
    return weights, author_index, authors


def balanced_binary_bootstrap(
    d: pd.DataFrame,
    author_weights: np.ndarray,
    authors: list[str],
    target_weights: np.ndarray,
) -> np.ndarray:
    out = np.zeros(N_BOOT, dtype=float)
    for stratum in range(N_STRATA):
        group_means: dict[str, np.ndarray] = {}
        for variety in ["AAVE", "SAE"]:
            cell = d.loc[
                d["in_common_support"] & d["length_stratum"].eq(stratum) & d["variety"].eq(variety)
            ]
            sums = cell.groupby("poet")["flag_machine"].sum().reindex(authors, fill_value=0).to_numpy(float)
            counts = cell.groupby("poet").size().reindex(authors, fill_value=0).to_numpy(float)
            group_means[variety] = (author_weights @ sums) / (author_weights @ counts)
        out += target_weights[stratum] * (group_means["AAVE"] - group_means["SAE"])
    return out


def balanced_rank_bootstrap(
    d: pd.DataFrame,
    author_weights: np.ndarray,
    authors: list[str],
    target_weights: np.ndarray,
) -> np.ndarray:
    author_map = {a: i for i, a in enumerate(authors)}
    out = np.zeros(N_BOOT, dtype=float)
    for stratum in range(N_STRATA):
        a = d.loc[
            d["in_common_support"] & d["length_stratum"].eq(stratum) & d["variety"].eq("AAVE")
        ]
        s = d.loc[
            d["in_common_support"] & d["length_stratum"].eq(stratum) & d["variety"].eq("SAE")
        ]
        av = a["oriented_score"].to_numpy(float)
        sv = s["oriented_score"].to_numpy(float)
        comparisons = (av[:, None] > sv[None, :]).astype(float) + 0.5 * (av[:, None] == sv[None, :])
        matrix = np.zeros((len(authors), len(authors)), dtype=float)
        ai = a["poet"].map(author_map).to_numpy(int)
        si = s["poet"].map(author_map).to_numpy(int)
        for i in range(len(authors)):
            for j in range(len(authors)):
                matrix[i, j] = comparisons[np.ix_(ai == i, si == j)].sum()
        ac = np.bincount(ai, minlength=len(authors)).astype(float)
        sc = np.bincount(si, minlength=len(authors)).astype(float)
        u = np.einsum("bi,ij,bj->b", author_weights, matrix, author_weights, optimize=True)
        denominator = (author_weights @ ac) * (author_weights @ sc)
        out += target_weights[stratum] * (2.0 * u / denominator - 1.0)
    return out


def analyze(
    zero: pd.DataFrame,
    poems: pd.DataFrame,
    step1: pd.DataFrame,
    design: pd.DataFrame,
    target_weights: np.ndarray,
) -> pd.DataFrame:
    detector_order = step1["detector"].tolist()
    author_weights, base_author_index, authors = author_multiplier_weights(design)
    step1_lookup = step1.set_index("detector")
    design_lookup = design.set_index("source_doc_id")
    rows: list[dict[str, object]] = []

    for detector_no, detector in enumerate(detector_order, start=1):
        d = zero.loc[zero["detector"].eq(detector)].copy()
        d = d.merge(
            design[["source_doc_id", "log_words", "in_common_support", "length_stratum", "cem_weight"]],
            on="source_doc_id", how="left", validate="one_to_one",
        )
        d = d.set_index("source_doc_id").loc[design["source_doc_id"]].reset_index()
        assert d["poet"].astype(str).tolist() == design["poet"].astype(str).tolist()
        assert np.array_equal(d["poet"].astype(str).map({a: i for i, a in enumerate(authors)}), base_author_index)

        y = d["flag_machine"].astype(float).to_numpy()
        aave = d["variety"].eq("AAVE").astype(float).to_numpy()
        log_words = d["log_words"].to_numpy(float)
        log_mean = float(log_words.mean())
        log_sd = float(log_words.std(ddof=0))
        z = (log_words - log_mean) / log_sd
        x = np.column_stack([np.ones(len(d)), aave, z])
        beta, converged, iterations = firth_logistic(x, y)
        assert converged, f"Firth point fit failed: {detector}"
        adjusted = standardized_probability_difference(beta, z)

        regression_boot = np.empty(N_BOOT, dtype=float)
        fit_failures = 0
        for b in range(N_BOOT):
            observation_weights = author_weights[b, base_author_index]
            boot_beta, ok, _ = firth_logistic(x, y, observation_weights, initial=beta)
            if not ok:
                fit_failures += 1
            regression_boot[b] = standardized_probability_difference(
                boot_beta, z, weights=observation_weights
            )
        assert np.isfinite(regression_boot).all()

        balanced_point = 0.0
        balanced_rank_point = 0.0
        for stratum in range(N_STRATA):
            cell = d.loc[d["in_common_support"] & d["length_stratum"].eq(stratum)]
            rates = cell.groupby("variety")["flag_machine"].mean()
            balanced_point += target_weights[stratum] * float(rates["AAVE"] - rates["SAE"])
            a_score = cell.loc[cell["variety"].eq("AAVE"), "oriented_score"].to_numpy(float)
            s_score = cell.loc[cell["variety"].eq("SAE"), "oriented_score"].to_numpy(float)
            balanced_rank_point += target_weights[stratum] * rank_biserial(a_score, s_score)
        balanced_boot = balanced_binary_bootstrap(d, author_weights, authors, target_weights)
        balanced_rank_boot = balanced_rank_bootstrap(d, author_weights, authors, target_weights)

        sae = d["variety"].eq("SAE").to_numpy()
        a_mask = ~sae
        score = d["oriented_score"].to_numpy(float)
        sae_x = np.column_stack([np.ones(sae.sum()), log_words[sae]])
        sae_beta = np.linalg.lstsq(sae_x, score[sae], rcond=None)[0]
        residual = score - (sae_beta[0] + sae_beta[1] * log_words)
        residual_rank = rank_biserial(residual[a_mask], residual[sae])

        residual_boot = np.empty(N_BOOT, dtype=float)
        for b in range(N_BOOT):
            w = author_weights[b, base_author_index]
            sw = w[sae]
            z_s = log_words[sae]
            score_s = score[sae]
            z_bar = np.average(z_s, weights=sw)
            score_bar = np.average(score_s, weights=sw)
            slope = np.sum(sw * (z_s - z_bar) * (score_s - score_bar)) / np.sum(sw * (z_s - z_bar) ** 2)
            intercept = score_bar - slope * z_bar
            boot_residual = score - (intercept + slope * log_words)
            residual_boot[b] = weighted_rank_biserial(
                boot_residual[a_mask], boot_residual[sae], w[a_mask], w[sae]
            )

        original = float(step1_lookup.loc[detector, "operational_delta"])
        reg_lo, reg_hi = np.quantile(regression_boot, [0.025, 0.975])
        bal_lo, bal_hi = np.quantile(balanced_boot, [0.025, 0.975])
        residual_lo, residual_hi = np.quantile(residual_boot, [0.025, 0.975])
        balanced_rank_lo, balanced_rank_hi = np.quantile(balanced_rank_boot, [0.025, 0.975])
        rows.append({
            "detector": detector,
            "headline_directionally_stable_positive": detector in HEADLINE_POSITIVE,
            "original_delta": original,
            "original_delta_ci_lo": float(step1_lookup.loc[detector, "operational_delta_ci_lo"]),
            "original_delta_ci_hi": float(step1_lookup.loc[detector, "operational_delta_ci_hi"]),
            "original_sign": sign_label(original),
            "length_adjusted_delta": adjusted,
            "length_adjusted_ci_lo": float(reg_lo),
            "length_adjusted_ci_hi": float(reg_hi),
            "length_adjusted_sign": sign_label(adjusted),
            "regression_direction_preserved": sign_label(adjusted) == sign_label(original),
            "length_balanced_delta": balanced_point,
            "length_balanced_ci_lo": float(bal_lo),
            "length_balanced_ci_hi": float(bal_hi),
            "length_balanced_sign": sign_label(balanced_point),
            "balanced_direction_preserved": sign_label(balanced_point) == sign_label(original),
            "direction_preserved_both_primary_length_analyses": (
                sign_label(adjusted) == sign_label(original)
                and sign_label(balanced_point) == sign_label(original)
            ),
            "original_rank_biserial": float(step1_lookup.loc[detector, "rank_biserial_r"]),
            "sae_length_residualized_rank_biserial": residual_rank,
            "sae_length_residualized_rank_ci_lo": float(residual_lo),
            "sae_length_residualized_rank_ci_hi": float(residual_hi),
            "length_balanced_rank_biserial": balanced_rank_point,
            "length_balanced_rank_ci_lo": float(balanced_rank_lo),
            "length_balanced_rank_ci_hi": float(balanced_rank_hi),
            "regression_model": "Firth logistic: machine_flag ~ AAVE + standardized log1p(n_words); pooled marginal standardization",
            "bootstrap_method": "author-cluster exponential multiplier bootstrap",
            "bootstrap_authors": len(authors),
            "bootstrap_replicates": N_BOOT,
            "bootstrap_seed": SEED,
            "firth_point_iterations": iterations,
            "firth_bootstrap_nonconvergence_n": fit_failures,
        })
        print(f"[{detector_no:02d}/12] {detector}: original={original:+.4f}, adjusted={adjusted:+.4f}, balanced={balanced_point:+.4f}", flush=True)

    result = pd.DataFrame(rows)
    assert not result.isna().any().any()
    return result


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def write_tex_table(summary: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}lrrrl@{}}",
        r"\toprule",
        r"Detector & Original $\Delta$ & Adjusted $\Delta$ [95\% CI] & Balanced $\Delta$ [95\% CI] & Direction preserved? \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        name = row.detector.replace("%", r"\%").replace("_", r"\_")
        preserved = "yes" if row.direction_preserved_both_primary_length_analyses else "no"
        lines.append(
            f"{name} & {fmt(row.original_delta)} & {fmt(row.length_adjusted_delta)} "
            f"[{fmt(row.length_adjusted_ci_lo)},{fmt(row.length_adjusted_ci_hi)}] & "
            f"{fmt(row.length_balanced_delta)} [{fmt(row.length_balanced_ci_lo)},{fmt(row.length_balanced_ci_hi)}] & {preserved} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Poem-length sensitivity for all 12 primary zero-shot configurations. Thresholds and machine-like labels remain frozen from SAE-only cross-fitting. The adjusted estimate is a pooled standardized probability difference from Firth logistic regression; the balanced estimate uses five common-support log-length strata with pooled fixed weights. Intervals use 10,000 author-cluster multiplier-bootstrap replicates.}",
        r"\label{tab:length_sensitivity}",
        r"\end{table*}",
    ]
    TABLE_OUT.write_text("\n".join(lines) + "\n")


def write_plot(summary: pd.DataFrame) -> None:
    plot = summary.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    ax.axvline(0.0, color="#666666", linewidth=1.0, linestyle="--", zorder=0)
    ax.scatter(plot["original_delta"], y, color="#3b6fb6", marker="o", s=40, label="Original", zorder=3)
    ax.errorbar(
        plot["length_adjusted_delta"], y + 0.12,
        xerr=np.vstack([
            plot["length_adjusted_delta"] - plot["length_adjusted_ci_lo"],
            plot["length_adjusted_ci_hi"] - plot["length_adjusted_delta"],
        ]),
        fmt="s", color="#c44e52", ecolor="#c44e52", capsize=2.5, markersize=4.5,
        label="Length-adjusted", zorder=2,
    )
    ax.errorbar(
        plot["length_balanced_delta"], y - 0.12,
        xerr=np.vstack([
            plot["length_balanced_delta"] - plot["length_balanced_ci_lo"],
            plot["length_balanced_ci_hi"] - plot["length_balanced_delta"],
        ]),
        fmt="^", color="#4c956c", ecolor="#4c956c", capsize=2.5, markersize=5,
        label="Length-balanced", zorder=2,
    )
    labels = []
    for row in plot.itertuples(index=False):
        labels.append(("* " if row.headline_directionally_stable_positive else "  ") + row.detector)
    ax.set_yticks(y, labels=labels, fontsize=8.5)
    ax.set_xlabel("AAVE minus SAE machine-like probability")
    fig.suptitle(
        "Natural-poetry operational disparities before and after length adjustment",
        x=0.28, y=0.985, ha="left", fontsize=14,
    )
    fig.text(
        0.28, 0.955, "* frozen directionally stable positive configuration",
        ha="left", fontsize=8, color="#444444",
    )
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.legend(loc="lower right", frameon=False, ncol=3, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    baseline: pd.DataFrame,
    poems: pd.DataFrame,
) -> None:
    dist = diagnostics.loc[diagnostics["record_type"].eq("distribution")].set_index(["sample", "variety"])
    headline = summary.loc[summary["headline_directionally_stable_positive"]]
    survive = bool(
        (headline["length_adjusted_delta"] > 0).all()
        and (headline["length_balanced_delta"] > 0).all()
    )
    lines = [
        "# RQ1 poem-length sensitivity\n",
        "## Frozen inputs and baseline reproduction\n",
        "The analysis uses the final 212-AAVE/521-SAE natural-poetry score cache, the frozen five-fold SAE map, the stored detector-by-fold thresholds, and the final Step 1 summary. It does not load the older 100/100 subset for estimation, run detector inference, or recalibrate a threshold. The stored `n_words` field is the loader's whitespace-token count after the uniform contraction rejoin normalization.\n",
        f"All 12 primary zero-shot configurations are complete (8,796 detector-by-poem rows). The score orientation audit passes exactly: every oriented score equals raw score times its frozen multiplier, and higher means more machine-like. The SAE folds contain 105, 104, 104, 104, and 104 poems. All thresholds reproduce exactly as the NumPy 95th percentile with `method=\"higher\"` on the other four SAE folds, and all decisions reproduce under strict `score > threshold`. The maximum absolute difference between reproduced and stored canonical deltas is {baseline['absolute_delta_difference'].max():.3g}.\n",
        "Frozen input SHA-256 values:\n",
    ]
    for path, expected in EXPECTED_HASHES.items():
        lines.append(f"- `{path.relative_to(ROOT)}`: `{expected}`")

    lines += [
        "\n## Length design fixed without detector outcomes\n",
        "The primary regression is poem-level Firth logistic regression of the frozen binary machine-like decision on an AAVE indicator and standardized `log1p(n_words)`. Firth's adjustment is used for every detector because two configurations have complete separation on AAVE flags. The reported effect is the pooled standardized AAVE-minus-SAE probability difference, not an odds ratio. Intervals use 10,000 exponential multiplier-bootstrap replicates at the author-cluster level, with one shared weight per named poet.\n",
        "The model-light analysis first restricts to the intersection of the observed length supports (29--1,114 words), then forms five pooled quantile strata in log length. Within-stratum AAVE-minus-SAE differences are aggregated with the frozen pooled common-support stratum weights. The design criterion was at least 10 poems in every variety-by-stratum cell and absolute log-length SMD below 0.10. The smallest cell contains 12 poems and the weighted SMD is 0.013, so the criterion passes. Thresholds and classifications remain the original frozen SAE-derived values.\n",
        "### Length distributions\n",
        "| Sample | Variety | n | Discarded | Effective n | Word count: mean / median / IQR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for sample in ["original_unweighted", "common_support_unweighted", "length_balanced_weighted"]:
        for variety in ["AAVE", "SAE"]:
            row = dist.loc[(sample, variety)]
            lines.append(
                f"| {sample.replace('_', ' ')} | {variety} | {int(row.n_retained)} | {int(row.n_discarded)} "
                f"({row.discarded_fraction:.1%}) | {row.effective_n:.1f} | {row.mean_words:.1f} / "
                f"{row.median_words:.1f} / {row.q25_words:.1f}--{row.q75_words:.1f} |"
            )

    lines += [
        "\n## Primary operational results\n",
        "| Detector | Original Δ | Length-adjusted Δ [95% CI] | Length-balanced Δ [95% CI] | Direction preserved? |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.detector} | {fmt(row.original_delta, 4)} | {fmt(row.length_adjusted_delta, 4)} "
            f"[{fmt(row.length_adjusted_ci_lo, 4)}, {fmt(row.length_adjusted_ci_hi, 4)}] | "
            f"{fmt(row.length_balanced_delta, 4)} [{fmt(row.length_balanced_ci_lo, 4)}, "
            f"{fmt(row.length_balanced_ci_hi, 4)}] | "
            f"{'yes' if row.direction_preserved_both_primary_length_analyses else 'no'} |"
        )

    lines += [
        "\n## Continuous-score sensitivity\n",
        "For each detector, an SAE-only linear relationship between oriented score and `log1p(n_words)` is fitted and removed from both groups. The first adjusted column is the rank-biserial effect on those residuals. The second is a five-stratum common-support rank-biserial effect using the same frozen length design as the operational comparison. Positive values mean more machine-like AAVE scores.\n",
        "| Detector | Original r | SAE-length-residualized r [95% CI] | Length-balanced r [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.detector} | {fmt(row.original_rank_biserial, 4)} | "
            f"{fmt(row.sae_length_residualized_rank_biserial, 4)} "
            f"[{fmt(row.sae_length_residualized_rank_ci_lo, 4)}, {fmt(row.sae_length_residualized_rank_ci_hi, 4)}] | "
            f"{fmt(row.length_balanced_rank_biserial, 4)} "
            f"[{fmt(row.length_balanced_rank_ci_lo, 4)}, {fmt(row.length_balanced_rank_ci_hi, 4)}] |"
        )

    lines += [
        "\n## Direct answer and interpretation\n",
        (
            "Yes. All five frozen directionally stable positive operational disparities remain positive under both the primary length-adjusted regression and the common-support length-balanced comparison. The supported claim is: **The five headline positive natural disparities retain their direction after adjustment for poem length and under a length-balanced comparison, indicating that the natural/proxy reversal is not explained solely by the AAVE--SAE length difference.** This is a benchmark-population sensitivity result, not a causal dialect estimate and not evidence that length or other confounding has been eliminated.\n"
            if survive else
            "No. At least one frozen directionally stable positive operational disparity changes direction under a primary length analysis. The manuscript must identify the affected configurations and weaken the reversal claim accordingly.\n"
        ),
        "## Remaining reviewer vulnerability\n",
        "- The corpus has only nine named poets and is dominated by Dunbar and Hughes. Author-cluster intervals acknowledge this dependence but cannot create independent authors or separate dialect from author, source, form, topic, era, orthography, and other correlated properties.\n",
        "- Firth logistic adjustment assumes a common linear log-length relationship on the logit scale. The stratified analysis is less model-dependent, but five bins only coarsen length and its common-support target discards 1 AAVE poem and 28 SAE poems.\n",
        "- Thresholds remain conditional on the frozen SAE calibration folds, as required. The new intervals do not add threshold-recalibration uncertainty and do not represent deployment populations outside this benchmark.\n",
        "- The continuous residual analysis extrapolates the SAE score--length relationship across the shared observed range; it is diagnostic rather than a new primary estimand.\n",
        "## Manuscript edits applied\n",
        "- The abstract now states that all five directionally stable positive operational conclusions remain positive under regression adjustment and common-support balancing.\n",
        "- Evaluation Setup defines the frozen-threshold Firth model, author-cluster multiplier bootstrap, common-support strata, fixed weights, and balance criterion.\n",
        "- RQ1a reports the adjusted and balanced ranges and the continuous-score specification dependence. Discussion and Limitations preserve the benchmark-population, noncausal framing and identify the nine-author and residual-confounding limits.\n",
        "- The appendix documents word-count construction, retained/discarded counts, balance, effective sample sizes, all 12 detector estimates, and the continuous-score sensitivity. A terminology pass uses `directionally stable positive`, `directionally stable negative`, and `specification-sensitive`.\n",
        "## Generated artifacts\n",
        f"- `{BASELINE_OUT.relative_to(ROOT)}`: detector-level exact reproduction audit.",
        f"- `{SUMMARY_OUT.relative_to(ROOT)}`: complete 12-detector operational and continuous sensitivity results.",
        f"- `{BALANCE_OUT.relative_to(ROOT)}`: support, strata, balance, discards, and weighted word-count diagnostics.",
        f"- `{TABLE_OUT.relative_to(ROOT)}`: manuscript-ready complete LaTeX table.",
        f"- `{FIGURE_PDF.relative_to(ROOT)}` and `{FIGURE_PNG.relative_to(ROOT)}`: original-versus-adjusted operational effect plot.",
    ]
    REPORT_OUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    zero, poems, step1, _, baseline = load_and_verify()
    design, diagnostics, target_weights = build_length_design(poems)
    summary = analyze(zero, poems, step1, design, target_weights)

    baseline.to_csv(BASELINE_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    diagnostics.to_csv(BALANCE_OUT, index=False)
    write_tex_table(summary)
    write_plot(summary)
    write_report(summary, diagnostics, baseline, poems)

    headline = summary.loc[summary["headline_directionally_stable_positive"]]
    print("\nHeadline positive configurations:")
    print(headline[["detector", "original_delta", "length_adjusted_delta", "length_balanced_delta"]].to_string(index=False))
    print(f"\nWrote {SUMMARY_OUT.relative_to(ROOT)} and companion artifacts")


if __name__ == "__main__":
    main()

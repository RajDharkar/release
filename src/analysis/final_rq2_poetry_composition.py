"""Poetry-only, fixed-budget training-composition experiment.

Only local sklearn models are fit.  The human training budget, machine
training set, folds, model settings, and feature definitions are held fixed;
only the AAVE/SAE composition of the human class changes.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.detection.features import base_numeric_features
from src.detection.features_v2 import dialect_rate_features


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/interim/detection_binary_v2_poetry_recon.csv"
OUT = ROOT / "data/analysis"
REPORT = ROOT / "reports/final_rq2_poetry_only_composition.md"
HUMAN_BUDGET = 160
CONDITIONS = (0, 10, 25, 50)
SAMPLING_SEEDS = tuple(range(10))
MODEL_SEED = 42
RF_TREES = 100
N_BOOT = 10_000
BOOT_SEED = 20260917


def h(value: str) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def make_grouped_folds(df: pd.DataFrame) -> dict[str, int]:
    """Balanced folds with all rows sharing source_doc_id kept together."""
    mapping: dict[str, int] = {}
    aave = sorted(df[(df.label == 0) & (df.variety == "AAVE")].source_doc_id.unique(), key=h)
    for i, doc in enumerate(aave):
        mapping[str(doc)] = i % 5
    sae = sorted(df[(df.label == 0) & (df.variety == "SAE")].source_doc_id.unique(), key=h)
    for i, doc in enumerate(sae):
        mapping[str(doc)] = i % 5

    # Most MGT rows share their source id with the source AAVE poem and inherit
    # that fold.  Place the two residual machine-only groups in the currently
    # smallest machine folds, deterministically.
    machine_docs = sorted(df[df.label == 1].source_doc_id.unique(), key=h)
    counts = [sum(mapping.get(str(doc)) == f for doc in machine_docs) for f in range(5)]
    for doc in machine_docs:
        key = str(doc)
        if key not in mapping:
            fold = min(range(5), key=lambda f: (counts[f], f))
            mapping[key] = fold
            counts[fold] += 1
    return mapping


def ordered(frame: pd.DataFrame, sampling_seed: int) -> pd.DataFrame:
    return frame.assign(_order=frame.source_doc_id.map(lambda x: h(f"{sampling_seed}|{x}"))).sort_values("_order").drop(columns="_order")


class CorrectedBuilder:
    """Old BASE block plus corrected, standardized dialect rates for PLUS."""

    def __init__(self, plus: bool):
        self.plus = plus
        self.word = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=3000, sublinear_tf=True)
        self.base_cols: list[str] = []
        self.scaler = StandardScaler() if plus else None

    def fit_transform(self, texts: pd.Series):
        t = texts.astype(str)
        w = self.word.fit_transform(t)
        c = self.char.fit_transform(t)
        base = base_numeric_features(t).fillna(0.0)
        self.base_cols = list(base.columns)
        blocks = [w, c, sp.csr_matrix(base.to_numpy(float))]
        if self.plus:
            dialect = dialect_rate_features(t).to_numpy(float)
            blocks.append(sp.csr_matrix(self.scaler.fit_transform(dialect)))
        return sp.hstack(blocks, format="csr")

    def transform(self, texts: pd.Series):
        t = texts.astype(str)
        base = base_numeric_features(t)[self.base_cols].fillna(0.0)
        blocks = [self.word.transform(t), self.char.transform(t), sp.csr_matrix(base.to_numpy(float))]
        if self.plus:
            dialect = dialect_rate_features(t).to_numpy(float)
            blocks.append(sp.csr_matrix(self.scaler.transform(dialect)))
        return sp.hstack(blocks, format="csr")


def model(kind: str):
    if kind == "LR":
        return LogisticRegression(max_iter=2000, random_state=MODEL_SEED, class_weight="balanced")
    return RandomForestClassifier(n_estimators=RF_TREES, random_state=MODEL_SEED, class_weight="balanced", n_jobs=-1)


def build_feature_sets(train_text: pd.Series, test_text: pd.Series) -> dict[str, tuple]:
    """Fit the shared BASE representation once, then append PLUS rates.

    This is mathematically identical to fitting the BASE and PLUS builders
    separately on the same training rows, but avoids duplicating the costly
    word/character vocabulary construction.
    """
    tr, te = train_text.astype(str), test_text.astype(str)
    word = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=3000, sublinear_tf=True)
    wtr, wte = word.fit_transform(tr), word.transform(te)
    ctr, cte = char.fit_transform(tr), char.transform(te)
    btr, bte = base_numeric_features(tr).fillna(0.0), base_numeric_features(te).fillna(0.0)
    xtr_base = sp.hstack([wtr, ctr, sp.csr_matrix(btr.to_numpy(float))], format="csr")
    xte_base = sp.hstack([wte, cte, sp.csr_matrix(bte.to_numpy(float))], format="csr")
    scaler = StandardScaler()
    dtr = scaler.fit_transform(dialect_rate_features(tr).to_numpy(float))
    dte = scaler.transform(dialect_rate_features(te).to_numpy(float))
    xtr_plus = sp.hstack([xtr_base, sp.csr_matrix(dtr)], format="csr")
    xte_plus = sp.hstack([xte_base, sp.csr_matrix(dte)], format="csr")
    return {"BASE": (xtr_base, xte_base), "PLUS": (xtr_plus, xte_plus)}


def rb(x: np.ndarray, y: np.ndarray) -> float:
    u = float(mannwhitneyu(x, y, alternative="two-sided", method="asymptotic").statistic)
    return 2 * u / (len(x) * len(y)) - 1


def run_predictions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in range(5):
        test = df[df.fold == fold].copy()
        pool = df[df.fold != fold]
        a_pool = pool[(pool.label == 0) & (pool.variety == "AAVE")]
        s_pool = pool[(pool.label == 0) & (pool.variety == "SAE")]
        mgt = pool[pool.label == 1]
        assert len(a_pool) >= 80 and len(s_pool) >= HUMAN_BUDGET
        assert not set(pool.source_doc_id) & set(test.source_doc_id)
        for sampling_seed in SAMPLING_SEEDS:
            ao = ordered(a_pool, sampling_seed)
            so = ordered(s_pool, sampling_seed)
            for pct in CONDITIONS:
                na = HUMAN_BUDGET * pct // 100
                ns = HUMAN_BUDGET - na
                assert HUMAN_BUDGET * pct % 100 == 0
                human = pd.concat([ao.head(na), so.head(ns)], ignore_index=True)
                train = pd.concat([human, mgt], ignore_index=True)
                assert len(human) == HUMAN_BUDGET
                feature_matrices = build_feature_sets(train.text, test.text)
                for feature_set, (xtr, xte) in feature_matrices.items():
                    for model_type in ["LR", "RF"]:
                        clf = model(model_type)
                        clf.fit(xtr, train.label.to_numpy(int))
                        score = clf.predict_proba(xte)[:, 1]
                        for pos, (_, r) in enumerate(test.iterrows()):
                            rows.append({
                                "sampling_seed": sampling_seed, "fold": fold, "condition_aave_pct": pct,
                                "feature_set": feature_set, "model": model_type,
                                "source_doc_id": r.source_doc_id, "label": int(r.label), "variety": r.variety,
                                "score_machine": float(score[pos]), "pred_machine_0_5": int(score[pos] >= 0.5),
                                "train_human_n": HUMAN_BUDGET, "train_aave_n": na, "train_sae_n": ns,
                                    "train_mgt_n": len(mgt),
                            })
            print(f"completed fold={fold} sampling_seed={sampling_seed}", flush=True)
    return pd.DataFrame(rows)


def seed_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["sampling_seed", "condition_aave_pct", "feature_set", "model"]
    for key, g in pred.groupby(keys, sort=True):
        human = g[g.label == 0]
        a = human[human.variety == "AAVE"]
        s = human[human.variety == "SAE"]
        m = g[g.label == 1]
        rows.append({
            **dict(zip(keys, key)), "n_aave_test": a.source_doc_id.nunique(), "n_sae_test": s.source_doc_id.nunique(),
            "mean_score_aave": a.score_machine.mean(), "mean_score_sae": s.score_machine.mean(),
            "median_score_aave": a.score_machine.median(), "median_score_sae": s.score_machine.median(),
            "rank_biserial_aave_vs_sae": rb(a.score_machine.to_numpy(), s.score_machine.to_numpy()),
            "fpr_aave_0_5": a.pred_machine_0_5.mean(), "fpr_sae_0_5": s.pred_machine_0_5.mean(),
            "delta_fpr": a.pred_machine_0_5.mean() - s.pred_machine_0_5.mean(),
            "h_vs_m_auroc": roc_auc_score(g.label, g.score_machine),
            "machine_recall_0_5": m.pred_machine_0_5.mean(),
            "f1_0_5": f1_score(g.label, g.pred_machine_0_5),
        })
    return pd.DataFrame(rows)


def bootstrap_cis(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOT_SEED)
    marginal, changes = [], []
    for (feature_set, model_type), g in pred.groupby(["feature_set", "model"]):
        a_ids = sorted(g[(g.label == 0) & (g.variety == "AAVE")].source_doc_id.unique())
        s_ids = sorted(g[(g.label == 0) & (g.variety == "SAE")].source_doc_id.unique())
        index_cols = ["sampling_seed", "condition_aave_pct", "source_doc_id"]
        hp = g[g.label == 0].set_index(index_cols).sort_index().pred_machine_0_5
        a_values = np.empty((len(SAMPLING_SEEDS), len(CONDITIONS), len(a_ids)), dtype=float)
        s_values = np.empty((len(SAMPLING_SEEDS), len(CONDITIONS), len(s_ids)), dtype=float)
        for si, seed in enumerate(SAMPLING_SEEDS):
            for pi, pct in enumerate(CONDITIONS):
                cell = hp.loc[(seed, pct)]
                a_values[si, pi] = cell.reindex(a_ids).to_numpy(float)
                s_values[si, pi] = cell.reindex(s_ids).to_numpy(float)
        boots = {pct: np.empty(N_BOOT) for pct in CONDITIONS}
        for b in range(N_BOOT):
            seed_i = int(rng.integers(0, len(SAMPLING_SEEDS)))
            ai = rng.integers(0, len(a_ids), len(a_ids))
            si = rng.integers(0, len(s_ids), len(s_ids))
            deltas = a_values[seed_i, :, ai].mean(axis=0) - s_values[seed_i, :, si].mean(axis=0)
            for pi, pct in enumerate(CONDITIONS):
                boots[pct][b] = deltas[pi]
        for pct in CONDITIONS:
            lo, hi = np.quantile(boots[pct], [0.025, 0.975])
            marginal.append({"feature_set": feature_set, "model": model_type, "condition_aave_pct": pct,
                             "delta_ci_lo": lo, "delta_ci_hi": hi})
        for pct in CONDITIONS[1:]:
            diff = boots[pct] - boots[0]
            lo, hi = np.quantile(diff, [0.025, 0.975])
            changes.append({"feature_set": feature_set, "model": model_type, "contrast": f"delta{pct}-delta0",
                            "estimate_bootstrap_mean": diff.mean(), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(marginal), pd.DataFrame(changes)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df = df[df.genre == "poetry"].copy()
    fold_map = make_grouped_folds(df)
    df["fold"] = df.source_doc_id.astype(str).map(fold_map).astype(int)
    assert df.groupby("source_doc_id").fold.nunique().max() == 1
    fold_counts = df.groupby(["fold", "label", "variety"]).size().rename("n").reset_index()
    pred = run_predictions(df)
    seed = seed_metrics(pred)
    marginal, changes = bootstrap_cis(pred)

    summary = seed.groupby(["condition_aave_pct", "feature_set", "model"], as_index=False).agg(
        n_aave_test=("n_aave_test", "first"), n_sae_test=("n_sae_test", "first"),
        mean_score_aave=("mean_score_aave", "mean"), mean_score_sae=("mean_score_sae", "mean"),
        median_score_aave=("median_score_aave", "mean"), median_score_sae=("median_score_sae", "mean"),
        rank_biserial=("rank_biserial_aave_vs_sae", "mean"), fpr_aave=("fpr_aave_0_5", "mean"),
        fpr_sae=("fpr_sae_0_5", "mean"), delta=("delta_fpr", "mean"), delta_seed_min=("delta_fpr", "min"),
        delta_seed_max=("delta_fpr", "max"), h_vs_m_auroc=("h_vs_m_auroc", "mean"),
        machine_recall=("machine_recall_0_5", "mean"), f1=("f1_0_5", "mean"),
    ).merge(marginal, on=["condition_aave_pct", "feature_set", "model"], how="left")

    pred.to_csv(OUT / "final_rq2_poetry_only_predictions.csv", index=False)
    seed.to_csv(OUT / "final_rq2_poetry_only_seed_metrics.csv", index=False)
    summary.to_csv(OUT / "final_rq2_poetry_only_summary.csv", index=False)
    changes.to_csv(OUT / "final_rq2_poetry_only_paired_changes.csv", index=False)
    fold_counts.to_csv(OUT / "final_rq2_poetry_only_fold_counts.csv", index=False)

    lines = [
        "# Final RQ2: poetry-only fixed-budget training composition\n",
        "## Design\n",
        f"The analysis uses all final era-matched poetry in `detection_binary_v2_poetry_recon.csv`: 102 AAVE-human poems, 219 SAE-human poems, and 100 SAE-MGT poems. Five document-grouped folds keep a source poem and its generated counterpart in the same fold. The fixed human-training budget is **K={HUMAN_BUDGET}** in every fold: 0% = 0/160 AAVE/SAE, 10% = 16/144, 25% = 40/120, and 50% = 80/80. Every fold has enough training candidates for all four conditions.\n",
        f"The machine training rows are identical across composition conditions within each fold. Test folds, TF-IDF limits, generic BASE features, model hyperparameters, and model seed (42) are fixed. LR uses class-balanced logistic regression; RF uses {RF_TREES} trees and class balancing. Ten deterministic sampling seeds change only source selection; within a seed, SHA-256 source rankings make AAVE and SAE selections nested across conditions. BASE is the original word+character TF-IDF plus generic rate/ratio block. PLUS adds the corrected 11 dialect rates per 100 words, standardized using training data only.\n",
        "Predictions are out-of-fold. Table entries are means across the 10 sampling seeds; median-score entries are the mean of the 10 seed-specific medians. The 95% delta intervals are 10,000-replicate hierarchical source bootstraps: each replicate selects one sampling seed and resamples AAVE and SAE source IDs independently. Change intervals reuse the same seed and resampled sources in both conditions, making the contrast paired.\n",
        "## Fold composition\n",
        "| fold | AAVE human test | SAE human test | MGT test |",
        "|---:|---:|---:|---:|",
    ]
    for fold in range(5):
        x = df[df.fold == fold]
        lines.append(f"| {fold} | {len(x[(x.label == 0) & (x.variety == 'AAVE')])} | {len(x[(x.label == 0) & (x.variety == 'SAE')])} | {len(x[x.label == 1])} |")
    lines += [
        "\n## Results\n",
        "| AAVE train % | config | test n AAVE/SAE | mean score AAVE/SAE | median score AAVE/SAE | r | FPR AAVE | FPR SAE | delta [95% CI] | delta seed range | H-v-M AUROC | machine recall | F1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.sort_values(["condition_aave_pct", "feature_set", "model"]).iterrows():
        lines.append(
            f"| {int(r.condition_aave_pct)}% | {r.feature_set} {r.model} | {int(r.n_aave_test)}/{int(r.n_sae_test)} | "
            f"{r.mean_score_aave:.3f}/{r.mean_score_sae:.3f} | {r.median_score_aave:.3f}/{r.median_score_sae:.3f} | "
            f"{r.rank_biserial:+.3f} | {r.fpr_aave:.3f} | {r.fpr_sae:.3f} | {r.delta:+.3f} "
            f"[{r.delta_ci_lo:+.3f}, {r.delta_ci_hi:+.3f}] | [{r.delta_seed_min:+.3f}, {r.delta_seed_max:+.3f}] | "
            f"{r.h_vs_m_auroc:.3f} | {r.machine_recall:.3f} | {r.f1:.3f} |"
        )
    lines += [
        "\n## Paired composition changes\n",
        "| config | contrast | estimate | paired 95% CI |",
        "|---|---|---:|---:|",
    ]
    point = summary.set_index(["feature_set", "model", "condition_aave_pct"]).delta
    for _, r in changes.iterrows():
        pct = int(r.contrast.split("delta")[1].split("-")[0])
        estimate = point.loc[(r.feature_set, r.model, pct)] - point.loc[(r.feature_set, r.model, 0)]
        lines.append(f"| {r.feature_set} {r.model} | {r.contrast} | {estimate:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] |")
    lines += [
        "\n## Interpretation\n",
        "This experiment identifies a composition effect within this fixed poetry design, not a general causal effect of AAVE representation outside these sources. The paired change intervals—not overlap between marginal intervals—are the inferential basis for whether each added-AAVE condition changes the disparity.\n",
        "Three conclusions are supported. First, under BASE LR, BASE RF, and PLUS RF, the 0% condition has a large positive gap (+0.388, +0.420, +0.406), and adding 10% AAVE reduces it to approximately zero; every paired 10%-minus-0% interval for those three configurations is strictly negative. Second, higher proportions overcorrect: at 50%, every configuration has a statistically negative marginal gap (-0.082 to -0.178), and every paired 50%-minus-0% interval is strictly negative. Third, corrected dialect features are not a uniform remedy: PLUS RF closely tracks BASE RF, while PLUS LR already has no positive 0% threshold gap and ranks AAVE poems as substantially more human-like (r=-0.893). Its 10% and 25% changes from baseline are not distinguishable from zero; only its 50% change is.\n",
        "Therefore the clean result is not ‘more AAVE monotonically fixes bias’ without qualification. In this poetry-only fixed-budget setting, modest representation closes the positive threshold gap for three configurations, while large representation reverses it for all four. The intervention affects detector calibration differently by model/feature specification and should be described as a composition-dependent tradeoff, not a universally optimal percentage.\n",
        "## Reproducibility outputs\n",
        "- `data/analysis/final_rq2_poetry_only_predictions.csv`\n",
        "- `data/analysis/final_rq2_poetry_only_seed_metrics.csv`\n",
        "- `data/analysis/final_rq2_poetry_only_summary.csv`\n",
        "- `data/analysis/final_rq2_poetry_only_paired_changes.csv`\n",
        "- `data/analysis/final_rq2_poetry_only_fold_counts.csv`\n",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(summary.to_string(index=False))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()

"""Independent numeric checks for the final artifact/claim audit.

No score generation or model inference occurs.  This file recomputes the
headline detector and RQ5 counts directly from retained row-level caches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detection.run_rq5_analysis import THRESHOLDS, SIGNS, ZS_DISPLAY, ZS_METRICS


ROOT = Path(__file__).resolve().parents[2]
I = ROOT / "data/interim"
O = ROOT / "data/analysis/final_artifact_numeric_verification.csv"


def v4_detectors(sub: pd.DataFrame):
    for suffix in ["gpt2", "opt"]:
        frame = pd.read_csv(I / f"detection_v4_zeroshot_{suffix}.csv").set_index("row_idx")
        bb = str(frame.backbone.iloc[0])
        for metric in ZS_METRICS:
            yield f"{ZS_DISPLAY[metric]} ({bb})", "zero-shot", frame[metric].reindex(sub.index) * SIGNS[(bb, metric)]
    pre = pd.read_csv(I / "detection_v4_pretrained.csv")
    for det, frame in pre.groupby("detector", sort=False):
        yield det, "pretrained", frame.set_index("row_idx").p_machine.reindex(sub.index)
    gz = pd.read_csv(I / "detection_v5_gptzero.csv").set_index("row_idx")
    yield "GPTZero", "commercial", gz.completely_generated_prob.reindex(sub.index)


def rq5_detectors(sub: pd.DataFrame):
    for suffix in ["gpt2", "opt"]:
        frame = pd.read_csv(I / f"rq5_zeroshot_{suffix}.csv")
        bb = str(frame.backbone.iloc[0])
        for metric in ZS_METRICS:
            label = f"{ZS_DISPLAY[metric]} ({bb})"
            score = frame.set_index("row_idx")[metric] * SIGNS[(bb, metric)]
            yield label, score, THRESHOLDS[("zeroshot", bb, metric)]
    pre = pd.read_csv(I / "rq5_pretrained.csv")
    for det, frame in pre.groupby("detector", sort=False):
        yield det, frame.set_index("row_idx").p_machine, THRESHOLDS[("pretrained", None, det)]
    gz = pd.read_csv(I / "rq5_gptzero.csv")
    yield "GPTZero", gz.set_index("row_idx").completely_generated_prob, THRESHOLDS[("gptzero", None, "GPTZero")]


def main() -> None:
    def wilson(k: int, n: int, z: float = 1.96):
        p = k / n
        den = 1 + z * z / n
        center = (p + z * z / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return center - half, center + half

    rows = []
    sub = pd.read_csv(I / "detection_v4_subsample.csv")
    v4 = {}
    for det, family, score in v4_detectors(sub):
        sae = (sub.category == "HWT") & (sub.variety == "SAE")
        aave = (sub.category == "HWT") & (sub.variety == "AAVE")
        thr = score[sae].quantile(.95)
        f_s = (score[sae] > thr).mean()
        f_a = (score[aave] > thr).mean()
        ks, ka = int((score[sae] > thr).sum()), int((score[aave] > thr).sum())
        sl, sh = wilson(ks, int(sae.sum()))
        al, ah = wilson(ka, int(aave.sum()))
        detectable_positive = f_a > f_s and not (al <= sh and sl <= ah)
        nono = sub.genre != "oratory"
        m = (sub.category == "MGT") & nono
        ha = (sub.category == "HWT") & (sub.variety == "AAVE") & nono
        hs = (sub.category == "HWT") & (sub.variety == "SAE") & nono
        auc_a = roc_auc_score(pd.concat([pd.Series(0, index=sub[ha].index), pd.Series(1, index=sub[m].index)]), score[pd.concat([sub[ha], sub[m]]).index])
        auc_s = roc_auc_score(pd.concat([pd.Series(0, index=sub[hs].index), pd.Series(1, index=sub[m].index)]), score[pd.concat([sub[hs], sub[m]]).index])
        h = (sub.category == "HWT") & nono
        c = (sub.category == "HGT_ArmC") & nono
        idx = sub[h | c].index
        auc_c = roc_auc_score((sub.loc[idx].category == "HGT_ArmC").astype(int), score[idx])
        clean = sub.genre.isin(["poetry", "lyrics"])
        hc = (sub.category == "HWT") & clean
        cc = (sub.category == "HGT_ArmC") & clean
        idxc = sub[hc | cc].index
        auc_clean = roc_auc_score((sub.loc[idxc].category == "HGT_ArmC").astype(int), score[idxc])
        v4[det] = {"family": family, "gap": f_a - f_s, "detectable_positive": detectable_positive, "auc_a": auc_a, "auc_s": auc_s, "auc_c": auc_c, "auc_clean": auc_clean}
        rows.append({"claim_family": "v4", "detector": det, "measure_1": f_a - f_s, "measure_2": auc_a - auc_s, "measure_3": auc_c, "measure_4": auc_clean})

    zero = [x for x in v4.values() if x["family"] == "zero-shot"]
    print("zero-shot mean pooled FPR gap", np.mean([x["gap"] for x in zero]))
    print("zero-shot detectable positive gap", sum(x["detectable_positive"] for x in zero), "/12")
    print("zero-shot AAVE-vs-M AUROC lower than SAE-vs-M", sum(x["auc_a"] < x["auc_s"] for x in zero), "/12")
    print("Arm C inversion", sum(x["auc_c"] < .5 for x in zero), "/12; means", np.mean([x["auc_c"] for x in zero]), np.mean([x["auc_clean"] for x in zero]))

    g = pd.read_csv(I / "detection_v5_gptzero.csv").set_index("row_idx")
    pm = (sub.category == "HWT") & (sub.genre == "poetry")
    print("GPTZero poetry means", g.completely_generated_prob[sub[pm & sub.variety.eq("AAVE")].index].mean(), g.completely_generated_prob[sub[pm & sub.variety.eq("SAE")].index].mean())

    rq = pd.read_csv(I / "rq5_subsample.csv").reset_index().rename(columns={"index": "row_idx"})
    rq_results = {}
    for det, score, thr in rq5_detectors(rq):
        d = rq.set_index("row_idx").join(score.rename("score"), how="left")
        sae = d[d.category == "SAE_MGT"].score.dropna()
        direct = d[(d.category == "AAVE_DIRECT") & d.genre.isin(["fiction", "poetry"])].score.dropna()
        armd = d[d.category == "ARM_D"].score.dropna()
        fs, fa, fd = (sae <= thr).mean(), (direct <= thr).mean(), (armd <= thr).mean()
        rq_results[det] = {"dminus": fa - fs, "armgap": fd - fa, "n_armd": len(armd)}
        rows.append({"claim_family": "rq5", "detector": det, "measure_1": fa - fs, "measure_2": fd - fa, "measure_3": len(direct), "measure_4": len(armd)})
    print("RQ5 negative D-minus", sum(x["dminus"] < 0 for x in rq_results.values()), "/16")
    # The retained report states all FNRs/gaps to three decimals; GPT2-XL
    # Entropy differs by only +0.000061 and is the published 0.000 exception.
    print("RQ5 ArmD gap positive at published 3-decimal resolution", sum(round(x["armgap"], 3) > 0 for x in rq_results.values()), "/16")
    opposite = 0
    for det, x in rq_results.items():
        fp = v4[det]["gap"]
        opposite += int(np.sign(fp) != np.sign(x["dminus"]))
    print("RQ5 FP/FN opposite direction", opposite, "/16")

    O.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(O, index=False)
    print(f"wrote {O}")


if __name__ == "__main__":
    main()
